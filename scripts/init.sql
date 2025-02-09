CREATE DATABASE IF NOT EXISTS agent_zero;
USE agent_zero;

-- 统一的对话记录表
CREATE TABLE IF NOT EXISTS chat_records (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    role_id VARCHAR(64) NOT NULL COMMENT '角色ID',
    chat_id VARCHAR(64) NOT NULL COMMENT '对话ID',
    user_id VARCHAR(64) NOT NULL COMMENT '用户ID',
    input TEXT NOT NULL COMMENT '用户输入',
    output TEXT NOT NULL COMMENT 'AI响应',
    query_text TEXT COMMENT '实际查询文本',
    remark TEXT COMMENT '备注信息',
    summary TEXT COMMENT '对话概要',
    raw_entity_memory JSON COMMENT '原始实体记忆',
    processed_entity_memory TEXT COMMENT '处理后的实体记忆',
    raw_history JSON COMMENT '原始历史消息',
    processed_history JSON COMMENT '处理后的历史消息',
    prompt TEXT COMMENT '完整提示词',
    agent_info JSON COMMENT 'Agent运行时信息',
    timestamp DATETIME NOT NULL COMMENT '对话时间',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    
    -- 索引
    INDEX idx_user_role (user_id, role_id) COMMENT '用户角色联合索引',
    INDEX idx_chat (chat_id) COMMENT '对话ID索引',
    INDEX idx_created_at (created_at) COMMENT '创建时间索引'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='对话记录表';

