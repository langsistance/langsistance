# AgenticSeek API Routes 文档

## 概述

AgenticSeek API Routes 是一个模块化的 FastAPI 应用程序，提供知识管理、工具管理和智能查询处理功能。API 采用 RESTful 设计原则，支持 JSON 数据交换格式。

## 项目结构

```
api_routes/
├── __init__.py                 # 包初始化文件
├── models.py                  # 数据模型定义
├── knowledge.py               # 知识管理 API
├── tools.py                  # 工具管理 API
├── query.py                  # 查询处理 API
├── knowledge_api_doc.md      # 知识管理 API 文档
├── tools_api_doc.md          # 工具管理 API 文档
├── query_api_doc.md          # 查询处理 API 文档
├── models_api_doc.md         # 数据模型文档
└── README.md                 # 总体文档（本文件）
```

## 快速开始

### 基础信息
- **基础URL**: `http://localhost:7777`
- **API版本**: v0.1.0
- **数据格式**: JSON
- **字符编码**: UTF-8

### 认证方式
目前 API 使用简单的用户ID识别机制，无需复杂的认证流程。

### API 路由前缀

| 模块 | 路由前缀 | 说明 |
|-----|----------|------|
| 知识管理 | `/` | 知识库增删改查 |
| 工具管理 | `/` | 工具管理和工具响应处理 |
| 查询处理 | `/` | 智能问答和系统状态 |

## 主要功能模块

### 1. 知识管理 (Knowledge Management)
**路由前缀**: `/`

提供完整的知识库管理功能：
- ✅ 创建知识记录
- ✅ 删除知识记录（软删除）
- ✅ 更新知识记录
- ✅ 查询个人知识记录
- ✅ 查询公开知识记录
- ✅ 复制知识记录

**特色功能**:
- 自动生成向量嵌入
- Redis 缓存支持
- 模糊搜索
- 权限控制

### 2. 工具管理 (Tools Management)
**路由前缀**: `/`

提供工具和相关知识的管理功能：
- ✅ 创建工具和知识记录（事务处理）
- ✅ 更新工具信息
- ✅ 删除工具记录（软删除）
- ✅ 查询个人工具记录
- ✅ 查询公开工具记录
- ✅ 工具请求处理
- ✅ 工具响应保存

**特色功能**:
- 事务性工具和知识创建
- Redis 工具请求缓存
- 工具执行状态跟踪

### 3. 查询处理 (Query Processing)
**路由前缀**: `/`

提供核心的智能问答功能：
- ✅ 智能查询处理
- ✅ 知识工具匹配
- ✅ 系统状态查询
- ✅ 任务控制（停止/状态检查）
- ✅ 截图功能
- ✅ 健康检查

**特色功能**:
- AI 代理集成
- 向量相似度搜索
- 实时状态轮询
- 多模态结果返回

## API 端点总览

### 知识管理端点
| 方法 | 路径 | 说明 |
|-----|------|------|
| POST | `/create_knowledge` | 创建知识记录 |
| POST | `/delete_knowledge` | 删除知识记录 |
| POST | `/update_knowledge` | 修改知识记录 |
| GET | `/query_knowledge` | 查询知识记录 |
| GET | `/query_public_knowledge` | 查询公开知识记录 |
| POST | `/copy_knowledge` | 复制知识记录 |

### 工具管理端点
| 方法 | 路径 | 说明 |
|-----|------|------|
| POST | `/create_tool_and_knowledge` | 创建工具和知识记录 |
| POST | `/update_tool` | 更新工具 |
| POST | `/delete_tool` | 删除工具 |
| GET | `/query_tools` | 查询工具记录 |
| GET | `/query_public_tools` | 查询公开工具记录 |
| POST | `/get_tool_request` | 获取工具请求 |
| POST | `/save_tool_response` | 保存工具响应 |

### 核心查询端点
| 方法 | 路径 | 说明 |
|-----|------|------|
| GET | `/latest_answer` | 获取最新答案 |
| POST | `/query` | 处理查询请求 |
| GET | `/screenshot` | 获取截图 |
| POST | `/find_knowledge_tool` | 根据问题查找相关知识和工具 |

### 系统管理端点
| 方法 | 路径 | 说明 |
|-----|------|------|
| GET | `/health` | 健康检查 |
| GET | `/is_active` | 检查系统是否活跃 |
| GET | `/stop` | 停止当前操作 |

## 使用示例

### 完整工作流程示例

#### 1. 创建工具和知识
```javascript
```javascript
const createToolAndKnowledge = async () => {
  const data = {
    // Tool 信息
    tool_userId: "user123",
    tool_title: "天气查询工具",
    tool_description: "获取指定城市的天气信息",
    tool_url: "https://api.weather.com",
    tool_push: 1,
    tool_public: true,
    tool_timeout: 30,
    tool_params: JSON.stringify({ api_key: "required" }),
    
    // Knowledge 信息
    knowledge_userId: "user123",
    knowledge_question: "如何查询天气？",
    knowledge_description: "使用天气API查询城市天气",
    knowledge_answer: "可以通过调用天气API获取实时天气信息...",
    knowledge_public: true,
    knowledge_embeddingId: 0,
    knowledge_model_name: "gpt-3.5-turbo",
    knowledge_params: "{}"
  };
  
  const response = await fetch('/create_tool_and_knowledge', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  
  return await response.json();
};
```
#### 2. 智能查询处理
```javascript
const processQuery = async (question) => {
  // 提交查询
  const queryResponse = await fetch('/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: question,
      query_id: `query_${Date.now()}`
    })
  });
  
  let result = await queryResponse.json();
  
  // 如果查询未完成，进行轮询
  while (result.done !== "true") {
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    const latestResponse = await fetch('/latest_answer');
    result = await latestResponse.json();
  }
  
  return result;
};
```

#### 3. 知识工具匹配
```javascript
const findRelevantTool = async (userId, question) => {
  const response = await fetch('/find_knowledge_tool', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      userId: userId,
      question: question,
      top_k: 5,
      similarity_threshold: 0.8
    })
  });
  
  return await response.json();
};
```

## 数据库设计

### 主要数据表

#### knowledge 表
```sql
CREATE TABLE knowledge (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id VARCHAR(50) NOT NULL,
    question VARCHAR(100) NOT NULL,
    description TEXT,
    answer TEXT,
    public TINYINT(1) DEFAULT 0,
    model_name VARCHAR(200),
    tool_id INT,
    params TEXT,
    status TINYINT(1) DEFAULT 1,
    embedding_id INT DEFAULT 0,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

#### tools 表
```sql
CREATE TABLE tools (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id VARCHAR(50) NOT NULL,
    title VARCHAR(100) NOT NULL,
    description TEXT,
    url VARCHAR(1000) NOT NULL,
    push TINYINT(1) DEFAULT 0,
    public TINYINT(1) DEFAULT 0,
    status TINYINT(1) DEFAULT 1,
    timeout INT DEFAULT 30,
    params TEXT,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

## Redis 缓存策略

### 缓存键命名规则
- 知识嵌入: `knowledge_embedding_{knowledge_id}`
- 工具请求: `tool_request_{query_id}_{user_id}`
- 工具响应: `tool_response_{query_id}_{user_id}`

### 缓存生命周期
- 知识嵌入: 持久化存储
- 工具请求/响应: 临时存储（建议设置过期时间）

## 错误处理

### 标准错误响应格式
```json
{
  "success": false,
  "message": "错误描述",
  "errors": ["详细错误信息1", "详细错误信息2"]
}
```

### HTTP 状态码规范
| 状态码 | 说明 | 使用场景 |
|--------|------|----------|
| 200 | 成功 | 操作成功完成 |
| 400 | 请求错误 | 参数验证失败、业务逻辑错误 |
| 403 | 权限不足 | 用户无权限访问/修改资源 |
| 404 | 资源未找到 | 记录不存在、文件不存在 |
| 429 | 请求过多 | 系统繁忙，有其他任务在处理 |
| 500 | 服务器错误 | 内部错误、数据库连接失败 |

## 性能优化

### 数据库优化
- 为常用查询字段添加索引
- 使用连接池管理数据库连接
- 实现软删除避免数据丢失

### 缓存优化
- Redis 缓存向量嵌入数据
- 临时数据设置合理过期时间
- 批量操作减少网络开销

### API 优化
- 分页查询避免大量数据传输
- 字段长度限制防止恶意请求
- 异步处理长时间任务

## 安全考虑

### 输入验证
- 严格的参数长度限制
- SQL 注入防护（参数化查询）
- XSS 防护（输入转义）

### 权限控制
- 用户只能操作自己的资源
- 公开资源的访问控制
- 敏感操作的权限校验

### 数据保护
- 敏感信息不记录日志
- 软删除保护数据安全
- 定期备份重要数据

## 部署配置

### 环境变量
```bash
# 数据库配置
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=password
MYSQL_DATABASE=copiioai_db

# Redis 配置
REDIS_URL=redis://localhost:6379/0

# 服务配置
BACKEND_PORT=7777
```

### Docker 配置
API 支持 Docker 容器化部署，自动检测容器环境并调整配置。

## 监控和日志

### 日志记录
- 所有 API 调用记录访问日志
- 错误详情记录到错误日志
- 数据库操作记录操作日志

### 健康检查
使用 `/health` 端点进行服务健康检查。

## 版本更新

### 版本规则
采用语义化版本控制 (Semantic Versioning)：
- 主版本号：不兼容的 API 修改
- 次版本号：向下兼容的功能性新增
- 修订号：向下兼容的问题修正

### 当前版本
- **API 版本**: v0.1.0
- **发布日期**: 2024-09-10
- **支持状态**: 开发版

## 贡献指南

### 开发规范
1. 遵循 PEP 8 Python 编码规范
2. 使用 Pydantic 进行数据验证
3. 编写完整的 API 文档
4. 添加适当的错误处理
5. 编写单元测试

### 提交流程
1. Fork 项目仓库
2. 创建功能分支
3. 编写代码和测试
4. 更新相关文档
5. 提交 Pull Request

## 联系信息

- **项目名称**: AgenticSeek API Routes
- **维护团队**: AgenticSeek Development Team
- **文档更新**: 2024-09-10

---

## 附录

### 相关文档链接
- [Knowledge API 详细文档](./knowledge_api_doc.md)
- [Tools API 详细文档](./tools_api_doc.md)
- [Query API 详细文档](./query_api_doc.md)
- [Data Models 详细文档](./models_api_doc.md)

### 第三方依赖
- **FastAPI**: Web 框架
- **Pydantic**: 数据验证
- **PyMySQL**: MySQL 数据库驱动
- **Redis**: 缓存服务
- **Uvicorn**: ASGI 服务器