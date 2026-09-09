-- W3 微信登录 users 表迁移（2026-09，deploy 前先在目标库执行）
-- 决策 D2: 微信用户挂 users.oauth_provider='wechat' / oauth_provider_id=unionid|openid,
-- 与邮箱/Google 同表同 user_id; firebase_uid/email 对微信行为 NULL。
--
-- 注意: 本文件只在"中国主库/目标库"执行一次(幂等写法, 可重复执行)。
-- MySQL 唯一索引允许多个 NULL, 故 email/firebase_uid 保留唯一索引即可兼容微信空值。

-- 1) firebase_uid 允许 NULL(微信行无 Firebase 账号)
ALTER TABLE users
    MODIFY COLUMN firebase_uid VARCHAR(128) NULL,
    ALGORITHM=INPLACE, LOCK=NONE;

-- 2) email 允许 NULL(微信行无邮箱)
ALTER TABLE users
    MODIFY COLUMN email VARCHAR(255) NULL,
    ALGORITHM=INPLACE, LOCK=NONE;

-- 3) 微信身份唯一键: (oauth_provider, oauth_provider_id)
--    先清可能的历史重复再建唯一索引(重复时先人工合并)
DELETE u1 FROM users u1
JOIN users u2
  ON u1.oauth_provider = u2.oauth_provider
 AND u1.oauth_provider_id = u2.oauth_provider_id
 AND u1.user_id > u2.user_id
WHERE u1.oauth_provider IS NOT NULL AND u1.oauth_provider_id IS NOT NULL;

ALTER TABLE users
    ADD UNIQUE KEY uq_oauth_provider_id (oauth_provider, oauth_provider_id),
    ALGORITHM=INPLACE, LOCK=NONE;

-- 4) 校验: 微信行可空字段应为 NULL, 唯一键生效
--    SELECT oauth_provider, oauth_provider_id, user_id FROM users
--      WHERE oauth_provider='wechat';
