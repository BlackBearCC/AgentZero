import os
import json
import time
import re
import requests
from datetime import datetime

# 豆包API配置
API_KEY = "c9ca6542-afed-4746-b451-b42c4b82b7a6"
API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"  # 完整的API端点URL

# 读取提示模板
def read_prompt_template(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# 解析对话批次
def parse_batches(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式按【数字】分割内容
    batches = re.split(r'【(\d+)】', content)
    
    # 处理分割结果，将编号与内容配对
    result = []
    for i in range(1, len(batches), 2):
        batch_id = batches[i]
        batch_content = batches[i+1].strip()
        if batch_content:  # 确保内容不为空
            result.append({"id": batch_id, "content": batch_content})
    
    return result

# 处理单个批次
def process_batch(batch, prompt_template):
    # 提取主体和客体
    user_match = re.search(r'"主体":"([^"]+)"', batch["content"])
    character_match = re.search(r'"客体":"([^"]+)"', batch["content"])
    
    user = user_match.group(1) if user_match else ""
    character = character_match.group(1) if character_match else ""
    
    # 替换提示模板中的变量
    prompt = prompt_template.replace("{{user}}", user).replace("{{character}}", character)
    
    # 构建API请求
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "ep-20250123090943-fbwr4",  # 替换为您的实际端点ID
        "messages": [
            {"role": "system", "content": "你是一个专业的用户画像分析助手。"},
            {"role": "user", "content": prompt + "\n\n下面是需要分析的对话内容：\n" + batch["content"]}
        ],
        "stream": True,
        "temperature": 0.1
    }
    
    try:
        # 修改SSL设置，禁用证书验证并增加超时时间
        response = requests.post(
            API_URL, 
            headers=headers, 
            json=data, 
            stream=True,
            verify=False,  # 禁用SSL证书验证
            timeout=60  # 增加超时时间到60秒
        )
        
        # 打印请求状态供调试
        print(f"请求状态码: {response.status_code}")
        
        if response.status_code == 200:
            full_response = ""
            print(f"处理批次 【{batch['id']}】:")
            
            # 处理流式响应
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        line = line[6:]  # 移除 "data: " 前缀
                        if line == "[DONE]":
                            break
                        try:
                            json_obj = json.loads(line)
                            content = json_obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                print(content, end="", flush=True)
                                full_response += content
                        except json.JSONDecodeError:
                            continue
            
            print("\n")  # 输出完成后换行
            return {"batch_id": batch["id"], "user": user, "character": character, "response": full_response}
        else:
            print(f"批次【{batch['id']}】处理失败: {response.status_code}, {response.text}")
            return {"batch_id": batch["id"], "error": f"HTTP错误: {response.status_code}"}
    
    except Exception as e:
        print(f"批次【{batch['id']}】处理出错: {str(e)}")
        return {"batch_id": batch["id"], "error": str(e)}

# 保存结果
def save_results(results, output_file="画像更新结果.json"):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到 {output_file}")

# 主函数
def main():
    prompt_template_path = "src/prompts/tool/user_info_1000token.txt"
    test_file_path = "tests/results/comparison/画像更新测试.txt"
    output_file = f"用户画像更新结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # 读取提示模板
    prompt_template = read_prompt_template(prompt_template_path)
    
    # 解析测试文件
    batches = parse_batches(test_file_path)
    print(f"共找到 {len(batches)} 个批次")
    
    # 处理每个批次
    results = []
    for batch in batches:
        result = process_batch(batch, prompt_template)
        results.append(result)
        time.sleep(1)  # 避免API请求过于频繁
    
    # 保存结果
    save_results(results, output_file)

if __name__ == "__main__":
    main()