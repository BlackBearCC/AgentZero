import pandas as pd
from fastapi import UploadFile
import io
from typing import List, Dict, Any

async def read_data_file(file: UploadFile) -> List[Dict[str, Any]]:
    """读取上传的CSV或Excel文件
    
    Args:
        file (UploadFile): 上传的文件对象
    
    Returns:
        List[Dict[str, Any]]: 解析后的数据列表
    """
    try:
        content = await file.read()
        # 根据文件扩展名选择不同的读取方法
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise ValueError("不支持的文件格式，请上传CSV或Excel文件")
        
        # 转换为字典列表
        data = df.to_dict('records')
        
        # 验证数据格式
        for item in data:
            if 'input' not in item:
                raise ValueError("数据格式错误：缺少'input'列")
        
        return data
        
    except Exception as e:
        raise ValueError(f"文件读取失败: {str(e)}") 