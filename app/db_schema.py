from app.db import db_cursor


SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS ai_personas (
        user_id BIGINT PRIMARY KEY,
        preset_name VARCHAR(100) NOT NULL,
        name VARCHAR(100) NOT NULL,
        identity TEXT NOT NULL,
        core_traits JSONB NOT NULL DEFAULT '[]'::jsonb,
        speech_style JSONB NOT NULL DEFAULT '{}'::jsonb,
        behavior_rules JSONB NOT NULL DEFAULT '[]'::jsonb,
        speech_habits JSONB NOT NULL DEFAULT '[]'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_user_memory (
        user_id BIGINT PRIMARY KEY,
        profile JSONB NOT NULL DEFAULT '{}'::jsonb,
        preferences JSONB NOT NULL DEFAULT '{}'::jsonb,
        relationship_rules JSONB NOT NULL DEFAULT '{}'::jsonb,
        entities JSONB NOT NULL DEFAULT '{}'::jsonb,
        interests JSONB NOT NULL DEFAULT '[]'::jsonb,
        projects JSONB NOT NULL DEFAULT '[]'::jsonb,
        long_term_notes JSONB NOT NULL DEFAULT '[]'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_memory_items (
        id BIGSERIAL PRIMARY KEY,
        user_id BIGINT NOT NULL,
        type VARCHAR(50) NOT NULL DEFAULT 'semantic',
        category VARCHAR(100) NOT NULL DEFAULT 'general',
        content TEXT NOT NULL,
        source_message TEXT,
        importance DOUBLE PRECISION NOT NULL DEFAULT 0.5,
        confidence DOUBLE PRECISION NOT NULL DEFAULT 0.8,
        sensitivity VARCHAR(50) NOT NULL DEFAULT 'normal',
        status VARCHAR(50) NOT NULL DEFAULT 'active',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        last_accessed_at TIMESTAMPTZ,
        access_count INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_ai_memory_items_user_status
    ON ai_memory_items (user_id, status)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_ai_memory_items_user_category
    ON ai_memory_items (user_id, category)
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_chat_sessions (
        id BIGSERIAL PRIMARY KEY,
        user_id BIGINT NOT NULL,
        title TEXT NOT NULL DEFAULT 'Новый чат',
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_ai_chat_sessions_user_updated
    ON ai_chat_sessions (user_id, updated_at DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_chat_messages (
        id BIGSERIAL PRIMARY KEY,
        session_id BIGINT NOT NULL
            REFERENCES ai_chat_sessions(id) ON DELETE CASCADE,
        user_id BIGINT NOT NULL,
        role VARCHAR(20) NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_ai_chat_messages_session
    ON ai_chat_messages (session_id, id)
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_summaries (
        session_id BIGINT PRIMARY KEY
            REFERENCES ai_chat_sessions(id) ON DELETE CASCADE,
        user_id BIGINT NOT NULL,
        summary_text TEXT NOT NULL DEFAULT '',
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_metrics (
        id BIGSERIAL PRIMARY KEY,
        user_id BIGINT NOT NULL,
        session_id BIGINT NOT NULL
            REFERENCES ai_chat_sessions(id) ON DELETE CASCADE,
        request_chars INTEGER NOT NULL,
        response_chars INTEGER NOT NULL,
        total_latency_ms INTEGER NOT NULL,
        model_name VARCHAR(100) NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_ai_metrics_user_created
    ON ai_metrics (user_id, created_at DESC)
    """,
)


def ensure_schema() -> None:
    with db_cursor(commit=True) as cursor:
        for statement in SCHEMA_STATEMENTS:
            cursor.execute(statement)
