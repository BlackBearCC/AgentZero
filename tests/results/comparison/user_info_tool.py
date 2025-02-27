import json
import time
import re
import requests
import csv
import os
from datetime import datetime
from tqdm import tqdm
import urllib3

"""
用户画像提取工具

使用说明:
1. 将 user_info_1000token.txt 和 画像更新测试.txt 放在与本脚本同一目录下
2. 运行脚本即可自动处理批次并生成用户画像
3. 结果将保存为JSON和CSV文件

支持的模型:
- doubao-1.5-pro: 更强大的模型，适合复杂场景
- doubao-1.5-lite: 轻量级模型，处理速度更快
"""

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 豆包API配置
API_KEY = "c9ca6542-afed-4746-b451-b42c4b82b7a6"
API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

# 豆包模型配置
MODELS = {
    "doubao-1.5-pro": "ep-20250123090943-fbwr4",
    "doubao-1.5-lite": "ep-20250124091428-qvld6"
}

# 选择要使用的模型
SELECTED_MODEL = "doubao-1.5-pro"

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
def process_batch(batch, prompt_template, profile=""):
    # 提取主体和客体
    user_match = re.search(r'"主体":"([^"]+)"', batch["content"])
    character_match = re.search(r'"客体":"([^"]+)"', batch["content"])
    
    user = user_match.group(1) if user_match else ""
    character = character_match.group(1) if character_match else ""
    
    # 获取当前时间，格式为：YYYY-MM-DD HH:MM
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # 替换提示模板中的变量
    prompt = prompt_template.replace("{{user}}", user)\
                           .replace("{{character}}", character)\
                           .replace("{{profile}}", profile)\
                           .replace("{{current_time}}", current_time)
    
    # 构建API请求
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": MODELS[SELECTED_MODEL],
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": batch["content"]}
        ],
        "stream": True,
        "temperature": 0.1
    }
    
    start_time = time.time()  # 记录开始时间
    
    try:
        # 修改SSL设置，禁用证书验证
        response = requests.post(
            API_URL, 
            headers=headers, 
            json=data, 
            stream=True,
            verify=False  # 禁用SSL证书验证
        )
        
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
            
            end_time = time.time()  # 记录结束时间
            processing_time = int((end_time - start_time) * 1000)  
            
            print("\n")  
            return {
                "batch_id": batch["id"], 
                "user": user, 
                "character": character, 
                "response": full_response,
                "system_prompt": prompt,
                "user_input": batch["content"],
                "extraction_time": current_time,
                "processing_time_ms": processing_time
            }
        else:
            print(f"批次【{batch['id']}】处理失败: {response.status_code}, {response.text}")
            return {"batch_id": batch["id"], "error": f"HTTP错误: {response.status_code}"}
    
    except Exception as e:
        print(f"批次【{batch['id']}】处理出错: {str(e)}")
        return {"batch_id": batch["id"], "error": str(e)}

# 保存结果到JSON
def save_results(results, output_file="画像更新结果.json"):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到 {output_file}")

# 保存用户画像到CSV
def save_profile_to_csv(result, profile_json, csv_file):
    file_exists = os.path.isfile(csv_file)
    try:
        profile_dict = json.loads(profile_json)
        
        row = {
            "批次ID": result["batch_id"],
            "原始JSON": profile_json,
            "系统提示词": result["system_prompt"],
            "用户输入": result["user_input"],
            "提取时间": result["extraction_time"],
            "处理耗时(ms)": result["processing_time_ms"],
        }
        
        for key, value in profile_dict.items():
            row[key] = value
        
        # 写入CSV
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ["批次ID","原始JSON",] + list(profile_dict.keys())+["系统提示词","用户输入","提取时间","处理耗时(ms)"]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            # 写入数据行
            writer.writerow(row)
        
        print(f"用户画像已保存到CSV: {csv_file}")
        return True
    
    except Exception as e:
        print(f"保存用户画像到CSV时出错: {str(e)}")
        return False

# 主函数
def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_template_path = os.path.join(script_dir, "记忆提炼大师24.3.txt")
    test_file_path = os.path.join(script_dir, "画像更新测试.txt")
    output_file = os.path.join(script_dir, f"用户画像更新结果_{SELECTED_MODEL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    csv_file = os.path.join(script_dir, f"用户画像变化记录_{SELECTED_MODEL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    # 读取提示模板
    prompt_template = read_prompt_template(prompt_template_path)
    
    # 解析测试文件
    batches = parse_batches(test_file_path)
    print(f"共找到 {len(batches)} 个批次")
    print(f"使用模型: {SELECTED_MODEL}")
    
    # 初始化用户画像
    profile = ""
    
    # 处理每个批次，使用tqdm显示进度
    results = []
    for batch in tqdm(batches, desc="处理批次进度"):
        result = process_batch(batch, prompt_template, profile)
        results.append(result)
        
        # 如果成功获取到了响应，更新用户画像
        if "response" in result and result["response"]:
            try:
                # 尝试从响应中提取JSON格式的用户画像
                json_start = result["response"].find("{")
                json_end = result["response"].rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = result["response"][json_start:json_end]
                    # 更新用户画像
                    profile = json_str
                    # 保存到CSV
                    save_profile_to_csv(result, json_str, csv_file)
            except Exception as e:
                print(f"更新用户画像时出错: {str(e)}")
        
        time.sleep(1)  # 避免API请求过于频繁
    
    # 保存结果
    save_results(results, output_file)

if __name__ == "__main__":
    main()