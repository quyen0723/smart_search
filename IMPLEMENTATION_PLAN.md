# KẾ HOẠCH TRIỂN KHAI SMART SEARCH V2.0
## Implementation Master Plan (Revised)

**Document Version:** 2.0 (Điều chỉnh sau đánh giá thực tế)
**Project Code:** SS-V2-MIGRATION
**Timeline:** 4 Tuần (2 Sprints × 2 tuần)
**Team:** 1 Senior Backend (Lead) + 1 Junior Dev
**Created:** 2025-12-28
**Last Updated:** 2025-12-28

---

## THAY ĐỔI SO VỚI V1.0

| Thay đổi | V1.0 | V2.0 | Lý do |
|----------|------|------|-------|
| Timeline | 6 tuần | **4 tuần** | Nhiều component đã có sẵn |
| Sprints | 3 | **2** | Gộp Sprint 2+3 |
| Total Hours | 242h | **158h** | Giảm 35% |
| Tasks | 32 | **22** | Loại bỏ tasks đã hoàn thành |

### Components Đã Có Sẵn (Không Cần Làm)

| Component | File | Trạng thái |
|-----------|------|------------|
| MeilisearchClient | `search/meilisearch_client.py` | Hoàn chỉnh |
| Docker Compose | `docker-compose.yml` | Hoàn chỉnh |
| IncrementalIndexer | `indexing/incremental.py` | Hoàn chỉnh |
| GraphPersistence | `graph/persistence.py` | Hoàn chỉnh |
| EmbeddingCache | `embedding/cache.py` | Hoàn chỉnh |
| HybridSearcher | `search/hybrid.py` | Hoàn chỉnh |

---

## PHẦN 1: TỔNG QUAN CHIẾN LƯỢC

### 1.1 Critical Path (Điều Chỉnh)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REVISED CRITICAL PATH                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  PRE-SPRINT: VERIFICATION (Day 0)                                    │  │
│  │  ├── docker-compose up -d meilisearch                                │  │
│  │  ├── Add meilisearch-python-sdk to dependencies                      │  │
│  │  └── Verify HybridSearcher works                                     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                           │                                                 │
│                           ▼                                                 │
│  ┌──────────────┐                              ┌──────────────┐            │
│  │ SPRINT 1     │                              │ SPRINT 2     │            │
│  │ Quick Wins   │ ────────────────────────────▶│ Integration  │            │
│  │ (Tuần 1-2)   │        GATE 1                │ (Tuần 3-4)   │            │
│  └──────────────┘   (Async I/O Done)           └──────────────┘            │
│         │                                              │                    │
│         │                                              │                    │
│         ▼                                              ▼                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Deliverables:                    │  Deliverables:                   │  │
│  │  - Feature Flags                  │  - Meilisearch Integration       │  │
│  │  - Pre-compiled Regex             │  - Reference Index               │  │
│  │  - LRU File Cache                 │  - Startup Recovery              │  │
│  │  - Async File I/O                 │  - Production Hardening          │  │
│  │  - Performance Logging            │  - Go-Live Checklist             │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ══════════════════════════════════════════════════════════════════════    │
│  TOTAL: 4 WEEKS (vs 6 weeks in V1.0)                                        │
│  ══════════════════════════════════════════════════════════════════════    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Sprint Goals (Điều Chỉnh)

| Sprint | Goal | Success Metric | Hours |
|--------|------|----------------|-------|
| **Pre-Sprint** | Verification - Xác nhận hệ thống có thể chạy Full Mode | Meilisearch running, HybridSearcher works | 4h |
| **Sprint 1** | Quick Wins - Async I/O, Caching, Feature Flags | `find_references` < 3s, No blocking I/O | 42h |
| **Sprint 2** | Integration - Kết nối SimpleIndexer với Meilisearch, Reference Index | `search` < 50ms, Restart < 10s | 68h |

### 1.3 Feature Flags

```python
# src/smart_search/config.py - Thêm vào
class FeatureFlags(BaseSettings):
    """Feature flags for gradual rollout."""

    model_config = SettingsConfigDict(env_prefix="FF_")

    use_meilisearch_search: bool = False      # Sprint 2 enable
    use_reference_index: bool = False          # Sprint 2 enable
    use_file_cache: bool = True                # Sprint 1 enable
    use_async_io: bool = True                  # Sprint 1 enable
    use_hybrid_searcher: bool = False          # Sprint 2 enable
```

---

## PHẦN 2: PRE-SPRINT VERIFICATION

**Mục đích:** Xác nhận codebase hiện tại có thể chạy được với Meilisearch

### Checklist Pre-Sprint

| Task | Command | Expected Result |
|------|---------|-----------------|
| Start Meilisearch | `docker-compose up -d meilisearch` | Container running |
| Check Meilisearch health | `curl http://localhost:7700/health` | `{"status":"available"}` |
| Add async SDK | Thêm `meilisearch-python-sdk` vào pyproject.toml | Import works |
| Start server | `python -m smart_search.main` | No errors |
| Check mode | Xem logs | "Full Mode" hoặc "SimpleIndexer" |

**Nếu Full Mode hoạt động:** Sprint 2 sẽ đơn giản hơn nhiều.
**Nếu SimpleIndexer:** Cần debug initialization trong Sprint 1.

---

## PHẦN 3: KẾ HOẠCH CHI TIẾT (WBS)

### SPRINT 1: QUICK WINS (Tuần 1-2)

**Sprint Goal:** Async I/O, Caching, Feature Flags, Performance baseline

| Task ID | Task Name | Owner | Priority | Hours | Dependencies | Acceptance Criteria |
|---------|-----------|-------|----------|-------|--------------|---------------------|
| **BE-101** | Add FeatureFlags to config.py | Junior | P0 | 2h | - | 5 flags defined, env vars work |
| **BE-102** | Pre-compile Regex patterns | Junior | P0 | 4h | - | Patterns in `__init__`, 30% CPU reduction |
| **BE-103** | Implement LRU File Cache | Senior | P0 | 4h | BE-101 | `@lru_cache` with mtime, hit ratio logged |
| **BE-104** | Add aiofiles + asyncio deps | Junior | P1 | 2h | - | pyproject.toml updated |
| **BE-105** | Create async file reader utility | Senior | P0 | 8h | BE-104 | `async_read_file()`, `async_read_files_parallel()` |
| **BE-106** | Refactor find_references to async | Senior | P0 | 12h | BE-103, BE-105 | Zero blocking, semaphore(50) |
| **BE-107** | Add performance logging middleware | Junior | P1 | 4h | - | Duration logged, >1s alerted |
| **BE-108** | Create benchmark suite | Junior | P1 | 4h | BE-107 | Automated tests for search, find_refs |
| **BE-109** | Debug Full Mode initialization | Senior | P1 | 4h | Pre-Sprint | Identify why SimpleIndexer is used |
| **BE-110** | Integration testing Sprint 1 | Senior | P0 | 4h | All above | All tests pass |

**Sprint 1 Total: 48h** (Senior: 32h, Junior: 16h)

**Sprint 1 Deliverables:**
```
✅ Feature flags system
✅ Pre-compiled regex (30% CPU reduction)
✅ LRU file cache (80%+ hit ratio)
✅ Async file I/O utilities
✅ find_references async (no blocking)
✅ Performance logging
✅ Benchmark suite
✅ Full Mode diagnosis
```

---

### SPRINT 2: INTEGRATION & HARDENING (Tuần 3-4)

**Sprint Goal:** Meilisearch integration, Reference Index, Production ready

| Task ID | Task Name | Owner | Priority | Hours | Dependencies | Acceptance Criteria |
|---------|-----------|-------|----------|-------|--------------|---------------------|
| **BE-201** | Fix orchestrator to use HybridSearcher | Senior | P0 | 8h | BE-109 | Full Mode works, not SimpleIndexer |
| **BE-202** | Create search router with feature flag | Senior | P0 | 6h | BE-201, BE-101 | FF controls Meili vs Legacy |
| **BE-203** | Implement dual-write in index_file() | Senior | P0 | 8h | BE-201 | Graph + Meilisearch sync |
| **BE-204** | Build index migration script | Junior | P1 | 6h | BE-203 | Backfill existing data |
| **BE-205** | Design Reference Index schema | Senior | P0 | 4h | - | Inverted index documented |
| **BE-206** | Implement ReferenceIndexer | Senior | P0 | 12h | BE-205 | O(1) lookup |
| **BE-207** | Integrate ReferenceIndexer in index_file | Senior | P0 | 6h | BE-206 | References indexed at parse |
| **BE-208** | Implement startup recovery | Senior | P0 | 6h | - | Load index from GraphPersistence |
| **BE-209** | Add Meilisearch health to /health | Junior | P1 | 2h | BE-201 | Health includes Meili status |
| **BE-210** | Load testing (100 users) | Junior | P0 | 6h | All above | Stable under load |
| **BE-211** | Security audit | Senior | P1 | 4h | - | No path traversal |
| **BE-212** | Final documentation | Junior | P1 | 6h | All above | API docs complete |
| **BE-213** | Go-Live verification | Senior | P0 | 4h | All above | All checklist green |

**Sprint 2 Total: 78h** (Senior: 58h, Junior: 20h)

**Sprint 2 Deliverables:**
```
✅ HybridSearcher active (not SimpleIndexer)
✅ Feature-flagged search routing
✅ Dual-write indexing
✅ Migration script
✅ Reference Index (O(1) lookup)
✅ Startup recovery from disk
✅ Health check includes Meilisearch
✅ Load tested (100 users)
✅ Security audited
✅ Documentation complete
```

---

## PHẦN 4: KIẾN TRÚC MỤC TIÊU

### 4.1 Current State vs Target State

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CURRENT STATE (SimpleIndexer Fallback)                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Request ──▶ FastAPI ──▶ SimpleIndexer ──▶ Brute-force O(n)                │
│                              │                                              │
│                              └──▶ Blocking file read                        │
│                                                                             │
│  Problems:                                                                  │
│  - search(): 500ms+ (O(n) scan)                                            │
│  - find_references(): 10s+ (read all files)                                │
│  - No persistence (lost on restart)                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

                                    │
                                    ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│  TARGET STATE (Full Mode with HybridSearcher)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐                                                            │
│  │   Request   │                                                            │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         FastAPI + Feature Flags                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│         │                                                                   │
│         ├─────────────────────┬─────────────────────┐                      │
│         ▼                     ▼                     ▼                      │
│  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐              │
│  │   /search   │       │ /references │       │   /index    │              │
│  └──────┬──────┘       └──────┬──────┘       └──────┬──────┘              │
│         │                     │                     │                      │
│         ▼                     ▼                     ▼                      │
│  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐              │
│  │ HybridSearch│       │ Reference   │       │   Smart     │              │
│  │ (Meilisearch│       │ Index       │       │  Indexer    │              │
│  │  <10ms)     │       │ (O(1) <1ms) │       │ (Async I/O) │              │
│  └──────┬──────┘       └──────┬──────┘       └──────┬──────┘              │
│         │                     │                     │                      │
│         └─────────────────────┴─────────────────────┘                      │
│                               │                                             │
│                               ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      PERSISTENCE LAYER                                │  │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐          │  │
│  │  │Meilisearch│  │  SQLite   │  │ LRU Cache │  │  Graph    │          │  │
│  │  │(search)   │  │(ref index)│  │(files)    │  │Persistence│          │  │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  Improvements:                                                              │
│  - search(): <10ms (Meilisearch)                                           │
│  - find_references(): <1ms (inverted index)                                │
│  - Persistent (survives restart)                                           │
│  - Async I/O (no blocking)                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## PHẦN 5: QUẢN TRỊ RỦI RO (Cập Nhật)

### 5.1 Risk Register

| Risk ID | Risk | Probability | Impact | Mitigation |
|---------|------|-------------|--------|------------|
| R-001 | Data inconsistency Meili vs Files | Medium | High | Dual-write, consistency job |
| R-002 | Legacy code regression | High | High | Feature flags, no direct edit |
| R-003 | Meilisearch downtime | Low | High | Fallback to legacy |
| R-004 | Re-index performance | Medium | Medium | Background job, rate limit |
| **R-007** | `meilisearch-python-sdk` missing | **100%** | High | Add to deps immediately |
| **R-008** | Full Mode init failure | High | High | Debug in Sprint 1 |
| **R-009** | HybridSearcher broken | Medium | High | Test before Sprint 2 |
| **R-010** | Async refactor breaks code | Medium | Medium | Incremental, test heavily |

### 5.2 Immediate Actions (Before Sprint 1)

```bash
# 1. Add missing dependency
cd /home/fong/Projects/smart_search
echo 'meilisearch-python-sdk>=0.31.0' >> requirements.txt
# Or add to pyproject.toml dependencies

# 2. Start Meilisearch
docker-compose up -d meilisearch

# 3. Verify
curl http://localhost:7700/health
# Expected: {"status":"available"}

# 4. Check current server mode
python -m smart_search.main
# Look for "SimpleIndexer" or "Full Mode" in logs
```

---

## PHẦN 6: GO-LIVE CHECKLIST

### 6.1 Performance Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| `search()` p50 | 500ms | < 20ms | ⬜ |
| `search()` p99 | 2000ms | < 50ms | ⬜ |
| `find_references()` p50 | 10s | < 100ms | ⬜ |
| `find_references()` p99 | 30s | < 500ms | ⬜ |
| Concurrent users | 4 | 100+ | ⬜ |
| Restart time | 20 min | < 10s | ⬜ |
| Memory | Unbounded | < 2GB | ⬜ |
| Cache hit ratio | 0% | > 80% | ⬜ |

### 6.2 Reliability Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Search accuracy vs legacy | >= 95% | ⬜ |
| Data persistence | 100% | ⬜ |
| Meilisearch health | Green | ⬜ |
| Rollback time | < 1 min | ⬜ |
| Blocking I/O | 0 | ⬜ |

### 6.3 Operational Items

| Item | Owner | Status |
|------|-------|--------|
| Meilisearch Docker ready | Junior | ⬜ |
| Runbook: Index rebuild | Junior | ⬜ |
| Runbook: Feature flag rollback | Senior | ⬜ |
| Monitoring configured | Senior | ⬜ |
| API docs updated | Junior | ⬜ |
| Load test passed | Junior | ⬜ |

---

## PHẦN 7: TIMELINE TỔNG HỢP

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         REVISED TIMELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Day 0          Week 1          Week 2          Week 3          Week 4     │
│    │              │               │               │               │         │
│    ▼              │               │               │               │         │
│  ┌────┐           │               │               │               │         │
│  │PRE │           │               │               │               │         │
│  │4h  │           │               │               │               │         │
│  └────┘           │               │               │               │         │
│    │              │               │               │               │         │
│    └──────────────┴───────────────┘               │               │         │
│           SPRINT 1 (48h)                          │               │         │
│           Quick Wins                              │               │         │
│           ├── Feature Flags                       │               │         │
│           ├── Async I/O                           │               │         │
│           ├── LRU Cache                           │               │         │
│           └── Performance Logging                 │               │         │
│                                   │               │               │         │
│                                   └───────────────┴───────────────┘         │
│                                          SPRINT 2 (78h)                     │
│                                          Integration                        │
│                                          ├── Meilisearch Active             │
│                                          ├── Reference Index                │
│                                          ├── Startup Recovery               │
│                                          └── Go-Live                        │
│                                                                             │
│  ════════════════════════════════════════════════════════════════════════  │
│  TOTAL: 4 weeks, 130 hours (vs 6 weeks, 242 hours in V1.0)                  │
│  SAVINGS: 2 weeks, 112 hours (46% reduction)                                │
│  ════════════════════════════════════════════════════════════════════════  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## PHẦN 8: EXISTING CODE TO LEVERAGE

### Files Đã Có - Chỉ Cần Tích Hợp

| File | Sử dụng cho | Cách tích hợp |
|------|-------------|---------------|
| `search/meilisearch_client.py` | BE-201, BE-203 | Import và sử dụng |
| `search/hybrid.py` | BE-201, BE-202 | Kích hoạt thay vì SimpleIndexer |
| `indexing/incremental.py` | BE-208 | Đã có sẵn |
| `graph/persistence.py` | BE-208 | Sử dụng cho startup recovery |
| `embedding/cache.py` | Reference cho SQLite | Pattern cho Reference Index |
| `docker-compose.yml` | Pre-Sprint | Chỉ cần `docker-compose up` |

### Files Cần Tạo Mới

| File | Task | Mô tả |
|------|------|-------|
| `utils/async_io.py` | BE-105 | Async file reading utilities |
| `search/reference_index.py` | BE-206 | Inverted index cho find_references |
| `api/search_router.py` | BE-202 | Feature flag routing |

---

**Document Version:** 2.0
**Prepared By:** Technical Project Manager
**Status:** Ready for Implementation
**Next Action:** Execute Pre-Sprint verification
