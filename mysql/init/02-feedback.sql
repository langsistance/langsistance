-- 用户反馈与消息通知系统表
USE copiioai_db;

-- 用户反馈表
CREATE TABLE feedback (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT UNSIGNED NOT NULL,
    email VARCHAR(255) NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    status TINYINT UNSIGNED NOT NULL DEFAULT 1 COMMENT '1:未读 2:已读 3:已回复',
    create_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    KEY idx_user_id (user_id),
    KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 消息表（管理员回复 / 系统通知）
CREATE TABLE messages (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT UNSIGNED NOT NULL,
    feedback_id BIGINT UNSIGNED DEFAULT NULL COMMENT '关联的反馈ID，可为空表示系统消息',
    title VARCHAR(256) NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    is_read TINYINT UNSIGNED NOT NULL DEFAULT 0 COMMENT '0:未读 1:已读',
    create_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_user_id_read (user_id, is_read),
    KEY idx_feedback_id (feedback_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
