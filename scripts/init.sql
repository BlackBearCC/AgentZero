CREATE DATABASE IF NOT EXISTS agent_zero;
USE agent_zero;

-- 统一的对话记录表
CREATE TABLE IF NOT EXISTS chat_records (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    role_id VARCHAR(50) NOT NULL COMMENT '角色ID',
    chat_id VARCHAR(50) NOT NULL COMMENT '对话ID',
    input_text TEXT NOT NULL COMMENT '用户输入',
    output_text TEXT NOT NULL COMMENT 'AI响应',
    summary TEXT COMMENT '对话概要',
    entity_memory JSON COMMENT '实体记忆',
    history_messages JSON COMMENT '历史对话消息',
    prompt TEXT COMMENT '完整提示词',
    llm_info JSON COMMENT 'LLM相关信息',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    
    -- 索引
    INDEX idx_role_chat (role_id, chat_id),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

