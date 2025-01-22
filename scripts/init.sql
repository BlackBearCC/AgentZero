CREATE DATABASE IF NOT EXISTS agent_zero;
USE agent_zero;

-- 对话记录表
CREATE TABLE IF NOT EXISTS chat_records (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    role_id VARCHAR(50) NOT NULL,
    chat_id VARCHAR(50) NOT NULL,
    input_text TEXT NOT NULL,
    output_text TEXT NOT NULL,
    summary TEXT,
    prompt TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_role_chat (role_id, chat_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 实体记忆表
CREATE TABLE IF NOT EXISTS entity_memories (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    chat_record_id BIGINT NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_data JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chat_record_id) REFERENCES chat_records(id),
    INDEX idx_chat_record (chat_record_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 对话历史表
CREATE TABLE IF NOT EXISTS chat_histories (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    chat_record_id BIGINT NOT NULL,
    history_data JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chat_record_id) REFERENCES chat_records(id),
    INDEX idx_chat_record (chat_record_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;