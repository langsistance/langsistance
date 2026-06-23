-- Long tasks persistence table
USE copiioai_db;

CREATE TABLE IF NOT EXISTS long_tasks (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    task_id         VARCHAR(64) NOT NULL UNIQUE COMMENT 'public task UUID',
    session_id      VARCHAR(64) DEFAULT NULL COMMENT 'associated session',
    user_id         BIGINT UNSIGNED NOT NULL,
    scene_id        BIGINT UNSIGNED DEFAULT NULL,
    task_type       VARCHAR(64) NOT NULL DEFAULT 'patent_analysis',
    input_params    JSON NOT NULL COMMENT 'query, patent_ids, model_family, etc.',
    status          VARCHAR(32) NOT NULL DEFAULT 'pending' COMMENT 'pending|running|completed|failed',
    progress        INT DEFAULT 0 COMMENT '0-100',
    current_phase   VARCHAR(64) DEFAULT NULL,
    current_step    VARCHAR(512) DEFAULT NULL,
    phases          JSON DEFAULT NULL COMMENT 'phase details with steps',
    table_columns   JSON DEFAULT NULL COMMENT 'Phase 1 output',
    table_rows      JSON DEFAULT NULL COMMENT 'Phase 2 output',
    result_summary  TEXT DEFAULT NULL COMMENT 'Phase 3 report text',
    report_files    JSON DEFAULT NULL COMMENT '[{format, path, size, created_at}]',
    error_message   TEXT DEFAULT NULL,
    celery_task_id  VARCHAR(128) DEFAULT NULL,
    create_time     DATETIME DEFAULT CURRENT_TIMESTAMP,
    update_time     DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    complete_time   DATETIME DEFAULT NULL,
    INDEX idx_user_id (user_id),
    INDEX idx_session_id (session_id),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
