"""Indexing API endpoints.

Provides endpoints for code indexing operations.
"""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/index", tags=["index"])


# Request/Response models
class IndexRequest(BaseModel):
    """Index request model."""

    paths: list[str] = Field(..., min_length=1, description="Paths to index")
    recursive: bool = Field(default=True, description="Recursively index directories")
    languages: list[str] | None = Field(default=None, description="Filter by languages")
    exclude_patterns: list[str] = Field(
        default_factory=lambda: ["**/node_modules/**", "**/.git/**", "**/__pycache__/**"],
        description="Glob patterns to exclude",
    )
    force_reindex: bool = Field(default=False, description="Force re-indexing of existing files")

    model_config = {
        "json_schema_extra": {
            "example": {
                "paths": ["/path/to/project"],
                "recursive": True,
                "languages": ["python", "javascript"],
            }
        }
    }


class IndexProgress(BaseModel):
    """Index progress model."""

    job_id: str
    status: str  # pending, running, completed, failed
    total_files: int
    processed_files: int
    indexed_files: int
    failed_files: int
    progress_percent: float
    current_file: str | None = None
    errors: list[str] = Field(default_factory=list)


class IndexResponse(BaseModel):
    """Index response model."""

    job_id: str
    status: str
    message: str


class FileInfo(BaseModel):
    """Indexed file info."""

    file_path: str
    language: str
    size_bytes: int
    last_modified: str
    code_units: int
    indexed_at: str | None = None


class IndexStats(BaseModel):
    """Index statistics."""

    total_files: int
    total_code_units: int
    languages: dict[str, int]
    files_by_type: dict[str, int]
    last_indexed: str | None = None
    index_size_bytes: int


class UpdateRequest(BaseModel):
    """Update index request."""

    paths: list[str] = Field(..., min_length=1, description="Paths to update")
    mode: str = Field(
        default="incremental",
        description="Update mode: incremental, full",
    )


class RemoveRequest(BaseModel):
    """Remove from index request."""

    paths: list[str] = Field(..., min_length=1, description="Paths to remove")


# Dependency placeholders
_indexer = None
_jobs: dict[str, IndexProgress] = {}


def get_indexer():
    """Get indexer instance."""
    if _indexer is None:
        raise HTTPException(
            status_code=503,
            detail="Indexer service not initialized",
        )
    return _indexer


def set_indexer(indexer):
    """Set indexer instance."""
    global _indexer
    _indexer = indexer


def get_jobs():
    """Get jobs dict."""
    return _jobs


# Helper functions
def _generate_job_id() -> str:
    """Generate unique job ID."""
    import uuid
    return str(uuid.uuid4())[:8]


def _validate_path(path: Path, base_paths: list[Path] | None = None) -> bool:
    """Validate path to prevent path traversal attacks.

    Args:
        path: Path to validate.
        base_paths: Allowed base directories (if None, just check for traversal).

    Returns:
        True if path is safe, False otherwise.
    """
    try:
        resolved = path.resolve()
        # Check for path traversal attempts
        if ".." in str(path):
            return False
        # If base paths specified, ensure resolved path is under one of them
        if base_paths:
            return any(
                resolved.is_relative_to(base.resolve())
                for base in base_paths
            )
        return True
    except (ValueError, OSError):
        return False


async def _run_index_job(
    job_id: str,
    paths: list[str],
    recursive: bool,
    languages: list[str] | None,
    exclude_patterns: list[str],
    force_reindex: bool,
    indexer,
):
    """Run indexing job in background."""
    jobs = get_jobs()
    jobs[job_id].status = "running"

    try:
        # Discover files with path validation
        all_files = []
        base_paths = [Path(p).resolve() for p in paths if Path(p).exists()]

        for path_str in paths:
            path = Path(path_str)

            # Security: validate path
            if not _validate_path(path):
                jobs[job_id].errors.append(f"Invalid path (security): {path_str}")
                continue

            if path.is_file():
                all_files.append(path.resolve())
            elif path.is_dir():
                if recursive:
                    for f in path.rglob("*"):
                        if _validate_path(f, base_paths):
                            all_files.append(f.resolve())
                else:
                    for f in path.glob("*"):
                        if _validate_path(f, base_paths):
                            all_files.append(f.resolve())

        # Filter files
        files_to_index = []
        for f in all_files:
            if not f.is_file():
                continue
            # Check exclude patterns
            import fnmatch
            excluded = any(
                fnmatch.fnmatch(str(f), pattern)
                for pattern in exclude_patterns
            )
            if excluded:
                continue
            # Check language filter
            if languages:
                ext = f.suffix.lower()
                lang = _ext_to_language(ext)
                if lang not in languages:
                    continue
            files_to_index.append(f)

        jobs[job_id].total_files = len(files_to_index)

        # Index files
        for i, file_path in enumerate(files_to_index):
            jobs[job_id].current_file = str(file_path)
            jobs[job_id].processed_files = i

            try:
                await indexer.index_file(file_path, force=force_reindex)
                jobs[job_id].indexed_files += 1
            except Exception as e:
                jobs[job_id].failed_files += 1
                jobs[job_id].errors.append(f"{file_path}: {str(e)}")

            jobs[job_id].progress_percent = (i + 1) / len(files_to_index) * 100 if files_to_index else 100

        jobs[job_id].processed_files = len(files_to_index)
        jobs[job_id].status = "completed"
        jobs[job_id].current_file = None

    except Exception as e:
        jobs[job_id].status = "failed"
        jobs[job_id].errors.append(f"Job failed: {str(e)}")


def _ext_to_language(ext: str) -> str:
    """Map file extension to language."""
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".cs": "csharp",
    }
    return mapping.get(ext, "unknown")


# Endpoints
@router.post("", response_model=IndexResponse)
async def start_index(
    request: IndexRequest,
    background_tasks: BackgroundTasks,
    indexer=Depends(get_indexer),
) -> IndexResponse:
    """Start indexing files.

    Starts a background job to index specified paths.
    Returns a job ID that can be used to track progress.
    """
    job_id = _generate_job_id()
    jobs = get_jobs()

    # Initialize job progress
    jobs[job_id] = IndexProgress(
        job_id=job_id,
        status="pending",
        total_files=0,
        processed_files=0,
        indexed_files=0,
        failed_files=0,
        progress_percent=0.0,
    )

    # Schedule background task
    background_tasks.add_task(
        _run_index_job,
        job_id=job_id,
        paths=request.paths,
        recursive=request.recursive,
        languages=request.languages,
        exclude_patterns=request.exclude_patterns,
        force_reindex=request.force_reindex,
        indexer=indexer,
    )

    return IndexResponse(
        job_id=job_id,
        status="pending",
        message=f"Indexing job started. Track progress at /index/progress/{job_id}",
    )


@router.get("/progress/{job_id}", response_model=IndexProgress)
async def get_progress(job_id: str) -> IndexProgress:
    """Get indexing job progress."""
    jobs = get_jobs()
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return jobs[job_id]


@router.post("/update", response_model=IndexResponse)
async def update_index(
    request: UpdateRequest,
    background_tasks: BackgroundTasks,
    indexer=Depends(get_indexer),
) -> IndexResponse:
    """Update index for specified paths.

    Performs incremental or full update of the index.
    """
    job_id = _generate_job_id()
    jobs = get_jobs()

    jobs[job_id] = IndexProgress(
        job_id=job_id,
        status="pending",
        total_files=0,
        processed_files=0,
        indexed_files=0,
        failed_files=0,
        progress_percent=0.0,
    )

    # For update, we use force_reindex based on mode
    force_reindex = request.mode == "full"

    background_tasks.add_task(
        _run_index_job,
        job_id=job_id,
        paths=request.paths,
        recursive=True,
        languages=None,
        exclude_patterns=["**/node_modules/**", "**/.git/**", "**/__pycache__/**"],
        force_reindex=force_reindex,
        indexer=indexer,
    )

    return IndexResponse(
        job_id=job_id,
        status="pending",
        message=f"Update job started ({request.mode} mode)",
    )


@router.post("/remove", response_model=dict[str, Any])
async def remove_from_index(
    request: RemoveRequest,
    indexer=Depends(get_indexer),
) -> dict[str, Any]:
    """Remove files/directories from index."""
    try:
        removed = 0
        errors = []

        for path_str in request.paths:
            try:
                count = await indexer.remove_path(path_str)
                removed += count
            except Exception as e:
                errors.append(f"{path_str}: {str(e)}")

        return {
            "removed_files": removed,
            "errors": errors,
            "status": "completed" if not errors else "completed_with_errors",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Remove failed: {str(e)}")


@router.get("/stats", response_model=IndexStats)
async def get_stats(
    indexer=Depends(get_indexer),
) -> IndexStats:
    """Get index statistics."""
    try:
        stats = await indexer.get_stats()

        return IndexStats(
            total_files=stats.get("total_files", 0),
            total_code_units=stats.get("total_code_units", 0),
            languages=stats.get("languages", {}),
            files_by_type=stats.get("files_by_type", {}),
            last_indexed=stats.get("last_indexed"),
            index_size_bytes=stats.get("index_size_bytes", 0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")


@router.get("/files", response_model=list[FileInfo])
async def list_indexed_files(
    pattern: str | None = Query(default=None, description="Glob pattern filter"),
    language: str | None = Query(default=None, description="Filter by language"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    indexer=Depends(get_indexer),
) -> list[FileInfo]:
    """List indexed files."""
    try:
        files = await indexer.list_files(
            pattern=pattern,
            language=language,
            limit=limit,
            offset=offset,
        )

        return [
            FileInfo(
                file_path=f.get("file_path", ""),
                language=f.get("language", "unknown"),
                size_bytes=f.get("size_bytes", 0),
                last_modified=f.get("last_modified", ""),
                code_units=f.get("code_units", 0),
                indexed_at=f.get("indexed_at"),
            )
            for f in files
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File listing failed: {str(e)}")


@router.get("/file/{file_path:path}", response_model=FileInfo | None)
async def get_file_info(
    file_path: str,
    indexer=Depends(get_indexer),
) -> FileInfo | None:
    """Get info for a specific indexed file."""
    try:
        info = await indexer.get_file_info(file_path)
        if not info:
            raise HTTPException(status_code=404, detail=f"File not indexed: {file_path}")

        return FileInfo(
            file_path=info.get("file_path", file_path),
            language=info.get("language", "unknown"),
            size_bytes=info.get("size_bytes", 0),
            last_modified=info.get("last_modified", ""),
            code_units=info.get("code_units", 0),
            indexed_at=info.get("indexed_at"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File info retrieval failed: {str(e)}")


@router.post("/refresh")
async def refresh_index(
    background_tasks: BackgroundTasks,
    indexer=Depends(get_indexer),
) -> dict[str, Any]:
    """Refresh index by checking for changes.

    Scans indexed files for modifications and updates the index accordingly.
    """
    job_id = _generate_job_id()
    jobs = get_jobs()

    jobs[job_id] = IndexProgress(
        job_id=job_id,
        status="pending",
        total_files=0,
        processed_files=0,
        indexed_files=0,
        failed_files=0,
        progress_percent=0.0,
    )

    async def refresh_task():
        jobs[job_id].status = "running"
        try:
            result = await indexer.refresh()
            jobs[job_id].indexed_files = result.get("updated", 0)
            jobs[job_id].status = "completed"
            jobs[job_id].progress_percent = 100.0
        except Exception as e:
            jobs[job_id].status = "failed"
            jobs[job_id].errors.append(str(e))

    background_tasks.add_task(refresh_task)

    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Refresh job started",
    }


@router.delete("/clear")
async def clear_index(
    confirm: bool = Query(..., description="Confirm clearing the index"),
    indexer=Depends(get_indexer),
) -> dict[str, Any]:
    """Clear all indexed data.

    This operation is irreversible. Use with caution.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. Set confirm=true to proceed.",
        )

    try:
        await indexer.clear()
        return {
            "status": "completed",
            "message": "Index cleared successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")


@router.get("/health")
async def health_check(
    indexer=Depends(get_indexer),
) -> dict[str, Any]:
    """Check index health.

    Returns the health status of the indexer and related services.
    """
    try:
        health = await indexer.health_check()
        return {
            "status": "healthy" if health.get("healthy", False) else "unhealthy",
            "services": health.get("services", {}),
            "issues": health.get("issues", []),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


class SimpleSearchResult(BaseModel):
    """Simple search result."""
    id: str
    name: str
    qualified_name: str
    code_type: str
    file_path: str
    line_start: int
    line_end: int
    score: float


class FileMatch(BaseModel):
    """A match found in a file."""
    line: int
    text: str
    type: str


class FileReference(BaseModel):
    """A file that references another file."""
    file_path: str
    language: str
    matches: list[FileMatch]
    match_count: int
    score: float


@router.get("/references")
async def find_references(
    filename: str = Query(..., min_length=1, description="Filename to find references for"),
    limit: int = Query(default=50, ge=1, le=200),
    indexer=Depends(get_indexer),
) -> list[FileReference]:
    """Find files that reference/use the given filename.

    Searches for:
    - PHP: require, require_once, include, include_once statements
    - Python: import, from...import statements
    - Function/class usage from the file

    Example: ?filename=de_get_users-helper.php
    """
    try:
        results = await indexer.find_references(filename, limit=limit)
        return [
            FileReference(
                file_path=r["file_path"],
                language=r["language"],
                matches=[FileMatch(**m) for m in r["matches"]],
                match_count=r["match_count"],
                score=r["score"],
            )
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reference search failed: {str(e)}")


@router.get("/search")
async def simple_search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=20, ge=1, le=100),
    language: str | None = Query(default=None, description="Filter by language (php, python)"),
    code_type: str | None = Query(default=None, description="Filter by type (function, class, method)"),
    indexer=Depends(get_indexer),
) -> list[SimpleSearchResult]:
    """Search indexed code units by name.

    Simple text-based search in code unit names and qualified names.
    """
    try:
        results = await indexer.search(q, limit=limit, language=language, code_type=code_type)
        return [SimpleSearchResult(**r) for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
