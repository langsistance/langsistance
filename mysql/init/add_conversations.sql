-- 会话持久化表
USE copiioai_db;

CREATE TABLE IF NOT EXISTS conversations (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    session_id      VARCHAR(64) NOT NULL UNIQUE COMMENT 'session unique identifier',
    user_id         BIGINT UNSIGNED NOT NULL COMMENT 'user ID from users table',
    scene_id        BIGINT UNSIGNED DEFAULT NULL COMMENT 'associated scene',
    title           VARCHAR(256) DEFAULT '' COMMENT 'session title, LLM-generated',
    messages        JSON NOT NULL COMMENT 'conversation messages array',
    long_task_ids   JSON DEFAULT NULL COMMENT 'linked long task IDs',
    status          TINYINT UNSIGNED DEFAULT 1 COMMENT '1=active, 2=archived',
    create_time     DATETIME DEFAULT CURRENT_TIMESTAMP,
    update_time     DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id),
    INDEX idx_create_time (create_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
