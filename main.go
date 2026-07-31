// Command memsteward is a demo agent: an AUTONOMOUS codebase STEWARD that keeps a
// living knowledge graph of a source tree in Jennah, consuming the public memory
// APIs exactly the way any external agent would — plain HTTP/JSON through the
// jennah-proxy gateway, authenticated with a jennah_sk_ API key. It is a
// standalone Go module (its own go.mod, not part of the server build) so it
// models a real outside consumer.
//
// memsteward runs UNATTENDED over a repo and exercises the parts of Jennah a
// long-running agent leans on:
//
//   - The EXECUTION LOG as durable per-file state. Each analyzed file writes a log
//     step carrying its content hash. On the next run the steward reads the log
//     back, so it re-analyzes only NEW or CHANGED files and detects REMOVED ones —
//     incremental work and drift detection, powered purely by remembered state.
//   - The GRAPH as structural knowledge. Files, the symbols they define, the
//     dependencies they import, and notable design decisions become nodes and
//     edges you can traverse ( repo → file → symbol/dep/decision ).
//   - VECTOR chunks for semantic recall of each file's summary.
//
// The split is deliberate: the LOG holds versioned state (hashes), the GRAPH holds
// relationships. Graph traversal rows don't reliably project node properties, but
// a log query returns every field of every step, so file hashes live in the log.
//
// Cross-run memory is simply reusing the same agent_instance_id, persisted to a
// small state file. Graph and log writes are keyed by deterministic content hashes,
// so re-running converges instead of fragmenting.
//
// The analysis brain is pluggable (see brain.go): Claude or Gemini, chosen by which
// API key is present or an explicit --provider. Only the LLM differs — every memory
// call is identical.
//
// Setup (Jennah key + one analysis provider). Keys come from env or flags:
//
//	export JENNAH_API_KEY=jennah_sk_...      # from POST /v1/apikeys
//	export ANTHROPIC_API_KEY=sk-ant-...      # Anthropic key, OR
//	export GEMINI_API_KEY=...                # Google AI Studio key
//	go run . -repo /path/to/repo             # analyze a tree; re-run to see it go incremental
//	go run . -show                           # print the remembered graph and exit
package main

import (
	"context"
	"crypto/rand"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"time"

	agentpb "github.com/alphauslabs/jennah-sdk-go/jennah/agent/v1"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"
)

// verbose makes the memory activity visible on screen: per-file classification and
// commit receipts. Set by --verbose.
var verbose bool

// repoNode is the stable anchor every file hangs off, so a traversal from it walks
// the whole structural graph. Created once, then never re-sent (nodes are
// insert-only server-side).
const repoNode = "repo"

// dirsToSkip are never walked; they hold vendored, generated, or VCS content.
var dirsToSkip = map[string]bool{
	".git": true, "vendor": true, "node_modules": true, "bin": true, ".idea": true, ".vscode": true,
}

func main() {
	var (
		endpoint     = flag.String("endpoint", envOr("JENNAH_ENDPOINT", "https://jennah.alphaus.cloud"), "Jennah proxy origin (http/https)")
		statePath    = flag.String("state", "memsteward-state.json", "path to the local state file (agent id)")
		provider     = flag.String("provider", "auto", "analysis LLM: auto|gemini|anthropic (auto prefers Anthropic, else Gemini, by which API key is set)")
		region       = flag.String("region", envOr("JENNAH_REGION", ""), "Jennah home region for the agent (e.g. us-central1); empty uses the platform default. Only applied when creating a new agent workspace. List regions with 'jnh agents regions'")
		jennahKey    = flag.String("jennah-api-key", "", "Jennah API key (jennah_sk_...); falls back to $JENNAH_API_KEY")
		anthropicKey = flag.String("anthropic-api-key", "", "Anthropic API key (sk-ant-...); falls back to $ANTHROPIC_API_KEY")
		repo         = flag.String("repo", ".", "path to the source tree to steward")
		ext          = flag.String("ext", ".go", "comma-separated file extensions to analyze (e.g. .go,.py,.ts)")
		maxFiles     = flag.Int("max-files", 40, "max files to consider in one run (0 = no limit)")
		show         = flag.Bool("show", false, "print the remembered codebase graph and exit (no analysis)")
	)
	flag.BoolVar(&verbose, "verbose", false, "print per-file classification and commit receipts")
	flag.Parse()

	// Fall back to the env vars, but keep them OUT of the flag defaults so --help
	// never prints the actual secret. A flag wins over its env var when both are set.
	*jennahKey = envOr2(*jennahKey, "JENNAH_API_KEY")
	*anthropicKey = envOr2(*anthropicKey, "ANTHROPIC_API_KEY")

	apiKey := *jennahKey
	if apiKey == "" {
		fatal("a Jennah API key is required: pass --jennah-api-key or set JENNAH_API_KEY (a jennah_sk_ key for an approved, entitled enterprise)")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	jc := &jennahClient{
		endpoint: strings.TrimRight(*endpoint, "/"),
		token:    apiKey,
		hc:       &http.Client{Timeout: 60 * time.Second},
	}

	st, err := loadState(*statePath)
	if err != nil {
		fatal("load state: %v", err)
	}

	repoAbs, err := filepath.Abs(*repo)
	if err != nil {
		fatal("resolve repo path: %v", err)
	}

	// One-time bootstrap: create the agent workspace and seed the stable "repo"
	// anchor node, then persist the id. Save only after the seed succeeds so a
	// crash mid-bootstrap doesn't leave a saved agent without its anchor node.
	if st.AgentID == "" {
		id, err := createAgent(ctx, jc, *region)
		if err != nil {
			fatal("create agent: %v", err)
		}
		if _, err := commit(ctx, jc, id, &agentpb.CommitMemoryRequest{
			AgentInstanceId: id,
			Graph:           &agentpb.GraphWrite{Nodes: []*agentpb.GraphNode{{NodeId: repoNode, Label: repoAbs}}},
		}); err != nil {
			fatal("seed repo node: %v", err)
		}
		st.AgentID = id
		save(*statePath, st)
		if *region != "" {
			fmt.Printf("created agent workspace %s (region %s)\n", id, *region)
		} else {
			fmt.Printf("created agent workspace %s (platform default region)\n", id)
		}
	} else {
		fmt.Printf("reusing agent workspace %s (memory carries over)\n", st.AgentID)
	}

	if *show {
		if err := showGraph(ctx, jc, st.AgentID); err != nil {
			fatal("show graph: %v", err)
		}
		return
	}

	// The one part that varies by provider: the analysis brain. Everything else is
	// provider-agnostic; the memory APIs don't care which LLM is thinking.
	br, err := newBrain(ctx, *provider, *anthropicKey)
	if err != nil {
		fatal("%v", err)
	}
	fmt.Printf("analysis model: %s\n", br.label())

	if err := runSteward(ctx, jc, br, st.AgentID, repoAbs, *ext, *maxFiles); err != nil {
		if errors.Is(err, context.Canceled) {
			fmt.Println("\ninterrupted — partial progress is saved in Jennah.")
			return
		}
		fatal("%v", err)
	}
}

// runSteward is one incremental pass over the tree: walk → diff against the
// remembered log → analyze only what changed → commit, and record drift.
func runSteward(ctx context.Context, jc *jennahClient, br brain, agentID, repoAbs, ext string, maxFiles int) error {
	files, truncated, err := walkRepo(repoAbs, ext, maxFiles)
	if err != nil {
		return fmt.Errorf("walk repo: %w", err)
	}
	if truncated {
		fmt.Printf("note: more than %d matching files; considering the first %d only\n", maxFiles, maxFiles)
	}
	fmt.Printf("scanning %s — %d file(s) matching %q\n", repoAbs, len(files), ext)

	known, err := recallFileState(ctx, jc, agentID)
	if err != nil {
		return fmt.Errorf("recall file state: %w", err)
	}

	onDisk := make(map[string]bool, len(files))
	var nNew, nMod, nSkip int
	for _, f := range files {
		onDisk[f.rel] = true
		k, seen := known[f.rel]
		switch {
		case !seen || k.removed:
			nNew++
			vlog("NEW      %s", f.rel)
			if err := analyzeAndCommit(ctx, jc, br, agentID, f, "new"); err != nil {
				return err
			}
		case k.hash != f.hash:
			nMod++
			vlog("MODIFIED %s", f.rel)
			if err := analyzeAndCommit(ctx, jc, br, agentID, f, "modified"); err != nil {
				return err
			}
		default:
			nSkip++
			vlog("unchanged %s (skipped)", f.rel)
		}
	}

	// Drift: files the log knows as present but that are gone from disk. Record the
	// drift in the log (graph is insert-only, so we don't tear nodes down).
	var removed []string
	for rel, k := range known {
		if k.removed || onDisk[rel] {
			continue
		}
		removed = append(removed, rel)
	}
	sort.Strings(removed)
	for _, rel := range removed {
		if err := recordDrift(ctx, jc, agentID, rel); err != nil {
			return err
		}
	}

	fmt.Printf("\nrun complete: %d new, %d modified, %d unchanged (skipped), %d removed\n", nNew, nMod, nSkip, len(removed))
	if len(removed) > 0 {
		fmt.Println("drift — files gone since last run:")
		for _, rel := range removed {
			fmt.Printf("  - %s\n", rel)
		}
	}
	fmt.Println("\ndone — run again after editing files to watch it go incremental, or `-show` the graph.")
	return nil
}

// analyzeAndCommit asks the brain to read one file, then writes the analysis to
// Jennah as a log step (carrying the content hash), a vector chunk, and graph
// nodes/edges — all in one atomic commit.
func analyzeAndCommit(ctx context.Context, jc *jennahClient, br brain, agentID string, f fileEntry, why string) error {
	content, err := readCapped(f.abs, 8000)
	if err != nil {
		return fmt.Errorf("read %s: %w", f.rel, err)
	}
	fa, err := br.analyzeFile(ctx, f.rel, content)
	if err != nil {
		return fmt.Errorf("analyze %s: %w", f.rel, err)
	}
	return commitFile(ctx, jc, agentID, f, fa, why)
}

// commitFile turns one fileAnalysis into memory. Node/edge ids are deterministic
// content hashes so re-committing converges. Dedup within the single commit: a
// mutation set can't carry two writes for the same key.
func commitFile(ctx context.Context, jc *jennahClient, agentID string, f fileEntry, fa fileAnalysis, why string) error {
	fileID := "f_" + hash(f.rel)
	nodes := []*agentpb.GraphNode{{NodeId: fileID, Label: f.rel, Properties: props(map[string]any{"lang": f.lang})}}
	edges := []*agentpb.GraphEdge{{
		EdgeId: "e_" + hash("CONTAINS|"+f.rel), SourceNodeId: repoNode, TargetNodeId: fileID, RelationshipType: "CONTAINS",
	}}
	seenN, seenE := map[string]bool{fileID: true}, map[string]bool{}

	addEdge := func(id, target, rel string) {
		if !seenE[id] {
			edges = append(edges, &agentpb.GraphEdge{EdgeId: id, SourceNodeId: fileID, TargetNodeId: target, RelationshipType: rel})
			seenE[id] = true
		}
	}
	addNode := func(id, label string, p map[string]any) {
		if !seenN[id] {
			nodes = append(nodes, &agentpb.GraphNode{NodeId: id, Label: label, Properties: props(p)})
			seenN[id] = true
		}
	}

	for _, s := range fa.Symbols {
		if strings.TrimSpace(s.Name) == "" {
			continue
		}
		sid := "s_" + hash(f.rel+"::"+s.Name)
		addNode(sid, s.Name, map[string]any{"kind": s.Kind})
		addEdge("e_"+hash("DEFINES|"+f.rel+"|"+s.Name), sid, "DEFINES")
	}
	for _, imp := range fa.Imports {
		if strings.TrimSpace(imp) == "" {
			continue
		}
		did := "d_" + hash(imp)
		addNode(did, imp, nil)
		addEdge("e_"+hash("IMPORTS|"+f.rel+"|"+imp), did, "IMPORTS")
	}
	for _, d := range fa.Decisions {
		if strings.TrimSpace(d.Title) == "" {
			continue
		}
		decID := "dec_" + hash(d.Title)
		addNode(decID, d.Title, map[string]any{"rationale": d.Rationale})
		addEdge("e_"+hash("NOTES|"+f.rel+"|"+d.Title), decID, "NOTES")
	}

	req := &agentpb.CommitMemoryRequest{
		AgentInstanceId: agentID,
		Log: &agentpb.ExecutionLogStep{
			StepId:         randID("step"),
			ThoughtProcess: f.hash, // per-file version marker, read back on the next run
			ToolUsed:       "analyze",
			ToolInput:      f.rel,
			ToolOutput:     truncate(fa.Summary, 1000),
		},
		Vectors: []*agentpb.VectorChunk{{
			ChunkId:    randID("chunk"),
			RawContent: f.rel + "\n" + fa.Summary,
		}},
		Graph: &agentpb.GraphWrite{Nodes: nodes, Edges: edges},
	}
	resp, err := commit(ctx, jc, agentID, req)
	if err != nil {
		return fmt.Errorf("commit %s: %w", f.rel, err)
	}
	fmt.Printf("  · %-8s %s  (%d symbols, %d imports, %d decisions)\n", why, f.rel, len(fa.Symbols), len(fa.Imports), len(fa.Decisions))
	printReceipt(resp, f.rel)
	return nil
}

// recordDrift writes a log step marking a file as gone. It is read back on the
// next run so we don't re-report the same removal every time.
func recordDrift(ctx context.Context, jc *jennahClient, agentID, rel string) error {
	_, err := commit(ctx, jc, agentID, &agentpb.CommitMemoryRequest{
		AgentInstanceId: agentID,
		Log: &agentpb.ExecutionLogStep{
			StepId: randID("step"), ThoughtProcess: "", ToolUsed: "drift", ToolInput: rel, ToolOutput: "removed",
		},
	})
	return err
}

// ---- recall (log + graph) ----

// fileMemory is the remembered state of one path, reconstructed from the log.
type fileMemory struct {
	hash    string
	removed bool
}

// recallFileState reads the execution log (newest first) and folds it into the
// latest known state per path: its content hash, or that it was last seen removed.
func recallFileState(ctx context.Context, jc *jennahClient, agentID string) (map[string]fileMemory, error) {
	var resp agentpb.QueryMemoryResponse
	if _, err := jc.do(ctx, http.MethodPost, memoryPath(agentID, "query"), &agentpb.QueryMemoryRequest{
		AgentInstanceId: agentID,
		Log:             &agentpb.LogQuery{Limit: 500},
	}, &resp); err != nil {
		return nil, err
	}
	// Steps come back newest first, so the first step we see for a path wins.
	out := map[string]fileMemory{}
	for _, s := range resp.GetLog().GetSteps() {
		rel := s.GetToolInput()
		if rel == "" {
			continue
		}
		if _, seen := out[rel]; seen {
			continue
		}
		switch s.GetToolUsed() {
		case "analyze":
			out[rel] = fileMemory{hash: s.GetThoughtProcess()}
		case "drift":
			out[rel] = fileMemory{removed: true}
		}
	}
	return out, nil
}

// showGraph prints the remembered structure: repo → files, then each file's
// symbols, imports, and decisions. Two one-hop traversals rather than one deep
// path, so every file is listed even if it has no children of a given kind.
func showGraph(ctx context.Context, jc *jennahClient, agentID string) error {
	fileRows, err := traverse(ctx, jc, agentID, &agentpb.GraphNodeMatch{
		Filters: []*agentpb.PropertyFilter{{Key: "NodeId", Value: structpb.NewStringValue(repoNode)}},
	}, "CONTAINS")
	if err != nil {
		return err
	}
	type fileRef struct{ id, label string }
	var fileRefs []fileRef
	for _, row := range fileRows {
		m := row.AsMap()
		if id := rowStr(m, "n1_id"); id != "" {
			fileRefs = append(fileRefs, fileRef{id: id, label: rowStr(m, "n1_label")})
		}
	}
	sort.Slice(fileRefs, func(i, j int) bool { return fileRefs[i].label < fileRefs[j].label })

	fmt.Printf("\ncodebase graph — %d file(s)\n", len(fileRefs))
	if len(fileRefs) == 0 {
		fmt.Println("(nothing remembered yet — run a scan first)")
		return nil
	}
	for _, fr := range fileRefs {
		fmt.Printf("\n▸ %s\n", fr.label)
		childRows, err := traverse(ctx, jc, agentID, &agentpb.GraphNodeMatch{
			Filters: []*agentpb.PropertyFilter{{Key: "NodeId", Value: structpb.NewStringValue(fr.id)}},
		}, "")
		if err != nil {
			return err
		}
		byRel := map[string][]string{}
		for _, row := range childRows {
			m := row.AsMap()
			rel, val := rowStr(m, "e0_type"), rowStr(m, "n1_label")
			if val != "" {
				byRel[rel] = append(byRel[rel], val)
			}
		}
		for _, rel := range []string{"DEFINES", "IMPORTS", "NOTES"} {
			vals := byRel[rel]
			if len(vals) == 0 {
				continue
			}
			sort.Strings(vals)
			fmt.Printf("    %s: %s\n", strings.ToLower(rel), strings.Join(vals, ", "))
		}
	}
	return nil
}

// traverse runs a start-node + single-hop OUTGOING query and returns the raw rows.
// relFilter, when non-empty, constrains the edge RelationshipType.
func traverse(ctx context.Context, jc *jennahClient, agentID string, start *agentpb.GraphNodeMatch, relFilter string) ([]*structpb.Struct, error) {
	var resp agentpb.QueryMemoryResponse
	if _, err := jc.do(ctx, http.MethodPost, memoryPath(agentID, "query"), &agentpb.QueryMemoryRequest{
		AgentInstanceId: agentID,
		Graph: &agentpb.GraphQuery{
			Start: start,
			Steps: []*agentpb.GraphStep{{
				Direction:        agentpb.GraphDirection_GRAPH_DIRECTION_OUTGOING,
				RelationshipType: relFilter,
				Node:             &agentpb.GraphNodeMatch{},
			}},
			Limit: 500,
		},
	}, &resp); err != nil {
		return nil, err
	}
	return resp.GetGraph().GetRows(), nil
}

// ---- repo walk ----

type fileEntry struct {
	rel  string // path relative to repo root
	abs  string
	hash string // content hash, the per-file version marker
	lang string // derived from extension
}

// walkRepo collects files under repoAbs whose extension is in ext (comma list),
// skipping VCS/vendor/hidden dirs, capped at maxFiles (0 = unlimited).
func walkRepo(repoAbs, ext string, maxFiles int) ([]fileEntry, bool, error) {
	exts := map[string]bool{}
	for _, e := range strings.Split(ext, ",") {
		e = strings.TrimSpace(e)
		if e == "" {
			continue
		}
		if !strings.HasPrefix(e, ".") {
			e = "." + e
		}
		exts[strings.ToLower(e)] = true
	}

	var out []fileEntry
	truncated := false
	err := filepath.WalkDir(repoAbs, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			name := d.Name()
			if path != repoAbs && (dirsToSkip[name] || strings.HasPrefix(name, ".")) {
				return fs.SkipDir
			}
			return nil
		}
		e := strings.ToLower(filepath.Ext(path))
		if !exts[e] {
			return nil
		}
		if maxFiles > 0 && len(out) >= maxFiles {
			truncated = true
			return fs.SkipAll
		}
		b, rerr := os.ReadFile(path)
		if rerr != nil {
			return rerr
		}
		rel, _ := filepath.Rel(repoAbs, path)
		out = append(out, fileEntry{rel: filepath.ToSlash(rel), abs: path, hash: hash(string(b)), lang: strings.TrimPrefix(e, ".")})
		return nil
	})
	if err != nil {
		return nil, false, err
	}
	sort.Slice(out, func(i, j int) bool { return out[i].rel < out[j].rel })
	return out, truncated, nil
}

func readCapped(path string, n int) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	if len(b) > n {
		b = b[:n]
	}
	return string(b), nil
}

// ---- agent + commit ----

// createAgent provisions the agent workspace. region is the optional Jennah home
// region ("" = platform default); it's honored only at creation time because an
// agent instance is pinned to one home region for its lifetime.
func createAgent(ctx context.Context, jc *jennahClient, region string) (string, error) {
	id := randID("agent")
	var resp agentpb.CreateAgentResponse
	if _, err := jc.do(ctx, http.MethodPost, "/v1/agents", &agentpb.CreateAgentRequest{
		AgentInstanceId: id,
		AgentName:       "memsteward-demo",
		Region:          region,
	}, &resp); err != nil {
		return "", err
	}
	if got := resp.GetAgent().GetAgentInstanceId(); got != "" {
		return got, nil
	}
	return id, nil
}

func commit(ctx context.Context, jc *jennahClient, agentID string, req *agentpb.CommitMemoryRequest) (*agentpb.CommitMemoryResponse, error) {
	var resp agentpb.CommitMemoryResponse
	_, err := jc.do(ctx, http.MethodPost, memoryPath(agentID, "commit"), req, &resp)
	return &resp, err
}

// printReceipt logs the commit counts and acts on the receipt's truncation report.
// rel is the repo-relative path of the file this commit summarized, used to name
// the offender (chunk ids here are random and identify nothing on their own).
func printReceipt(r *agentpb.CommitMemoryResponse, rel string) {
	ts := "?"
	if t := r.GetCommitTimestamp(); t != nil {
		ts = t.AsTime().UTC().Format(time.RFC3339)
	}
	vlog("committed: log=%d vec=%d nodes=%d edges=%d @ %s",
		r.GetExecutionLogRows(), r.GetVectorRows(), r.GetGraphNodeRows(), r.GetGraphEdgeRows(), ts)

	// The commit SUCCEEDED, but the embedding model truncated content past its
	// input limit (~2048 tokens): the file's summary is stored in full while its
	// vector covers only the beginning.
	//
	// Worth being precise about the blast radius here: memsteward itself never runs
	// a semantic query — it reads the execution log for per-file state and the graph
	// for structure — so truncation does NOT degrade anything this tool does. What
	// it degrades is the searchable map memsteward exists to BUILD, for whoever
	// queries it later (a person, or another agent). That is still worth reporting,
	// but it is a quality problem in the output rather than a malfunction in the run.
	//
	// Printed unconditionally, NOT under vlog, for the same reason as the sibling
	// demos: a silently partial index is not something a caller should have to opt in
	// to hearing about.
	//
	// reject_on_truncation is deliberately NOT set: aborting would lose the file's
	// graph nodes and its per-file state marker too, which would make the next run
	// re-analyze it and hit the same wall. Splitting a long file's summary into
	// several chunks is the real fix.
	if ids := r.GetTruncatedChunkIds(); len(ids) > 0 {
		fmt.Fprintf(os.Stderr, "[memory warning] summary of %s too long to embed in full, "+
			"semantic search over this file will only see its beginning (chunk %s)\n",
			rel, strings.Join(ids, ", "))
	}
}

// ---- HTTP client (protojson over the gateway, Bearer auth) ----

type jennahClient struct {
	endpoint string
	token    string
	hc       *http.Client
}

func (c *jennahClient) do(ctx context.Context, method, path string, in, out proto.Message) (int, error) {
	var body io.Reader
	if in != nil {
		b, err := protojson.Marshal(in)
		if err != nil {
			return 0, err
		}
		body = strings.NewReader(string(b))
	}
	req, err := http.NewRequestWithContext(ctx, method, c.endpoint+path, body)
	if err != nil {
		return 0, err
	}
	if in != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	req.Header.Set("Authorization", "Bearer "+c.token)

	resp, err := c.hc.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode >= 300 {
		return resp.StatusCode, fmt.Errorf("%s %s: HTTP %d: %s", method, path, resp.StatusCode, gatewayMessage(raw))
	}
	if out != nil {
		if err := (protojson.UnmarshalOptions{DiscardUnknown: true}).Unmarshal(raw, out); err != nil {
			return resp.StatusCode, fmt.Errorf("decode %s: %w", path, err)
		}
	}
	return resp.StatusCode, nil
}

func agentPath(id string) string        { return "/v1/agents/" + url.PathEscape(id) }
func memoryPath(id, verb string) string { return agentPath(id) + "/memory:" + verb }

func gatewayMessage(body []byte) string {
	var gw struct {
		Message string `json:"message"`
	}
	if err := json.Unmarshal(body, &gw); err == nil && strings.TrimSpace(gw.Message) != "" {
		return strings.TrimSpace(gw.Message)
	}
	return strings.TrimSpace(string(body))
}

// ---- local state ----

// state persists only the agent id — that's what makes memory carry across runs.
// No graph/log id tracking is needed: writes are keyed by deterministic content
// hashes, so re-running converges.
type state struct {
	AgentID string `json:"agent_id"`
}

func loadState(path string) (*state, error) {
	st := &state{}
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return st, nil
	}
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(b, st); err != nil {
		return nil, err
	}
	return st, nil
}

func save(path string, st *state) {
	b, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "warning: marshal state: %v\n", err)
		return
	}
	if err := os.WriteFile(path, b, 0o600); err != nil {
		fmt.Fprintf(os.Stderr, "warning: write state %s: %v\n", path, err)
	}
}

// ---- helpers ----

func props(m map[string]any) *structpb.Struct {
	if len(m) == 0 {
		return nil
	}
	clean := map[string]any{}
	for k, v := range m {
		if s, ok := v.(string); ok && strings.TrimSpace(s) == "" {
			continue
		}
		clean[k] = v
	}
	if len(clean) == 0 {
		return nil
	}
	s, err := structpb.NewStruct(clean)
	if err != nil {
		return nil
	}
	return s
}

func hash(s string) string {
	sum := sha1.Sum([]byte(s))
	return hex.EncodeToString(sum[:])[:12]
}

func randID(prefix string) string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return prefix + "_" + hex.EncodeToString(b[:])
}

func rowStr(m map[string]any, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// envOr2 returns val when it's non-empty (a flag was passed), otherwise the value
// of env var key. Unlike envOr, the caller-supplied value wins — so an explicit
// flag overrides the env var, and the env var is never a flag default (keeping
// secrets out of --help).
func envOr2(val, key string) string {
	if strings.TrimSpace(val) != "" {
		return val
	}
	return os.Getenv(key)
}

// vlog prints a dim diagnostic line to stdout, only when --verbose is set.
func vlog(format string, args ...any) {
	if verbose {
		fmt.Printf("  \033[2m%s\033[0m\n", fmt.Sprintf(format, args...))
	}
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "memsteward: "+format+"\n", args...)
	os.Exit(1)
}
