# CopiioAI 微信小程序（Taro + React + TS）

服务国内微信用户（MIGRATION_PLAN P5）。M1 = 微信登录 + 会话列表。

## 本地开发

```bash
npm install

# 后端本地起在 7777 时直接联调（开发者工具勾选"不校验合法域名"）：
npm run dev:weapp
# 或指定后端地址：
TARO_APP_API_BASE=https://your-gateway npm run dev:weapp
```

微信开发者工具导入本目录（project.config.json 已指向 `dist/`）。

## 待办（依赖用户侧资产）

- [ ] `project.config.json` 的 `appid`：换真实小程序 AppID（当前 touristappid）
- [ ] 后端部署 W3（/auth/wechat + users ALTER：`deploy/china/w3_users_wechat_migration.sql`）
- [ ] `.env` 加 `WECHAT_APPID` / `WECHAT_SECRET`
- [ ] 正式环境 API 基址：编译传 `TARO_APP_API_BASE`（备案域名或 AnyService 网关）

## 目录

```
src/
├── config.ts        # API 基址/存储键单一出口
├── services/        # api.ts(请求封装+401) auth.ts(微信登录) sessions.ts
└── pages/
    ├── index/       # 会话列表（首页，DeepSeek-app 式）
    ├── login/       # 微信一键登录（M1）
    └── chat/        # 对话页占位（M2）
```
