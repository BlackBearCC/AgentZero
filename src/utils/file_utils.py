import pandas as pd
from fastapi import UploadFile
import io
from typing import List, Dict, Any

async def read_data_file(file: UploadFile) -> Dict[str, Any]:
    """读取上传的CSV或Excel文件
    
    Args:
        file (UploadFile): 上传的文件对象
    
    Returns:
        Dict[str, Any]: 包含数据和字段信息的字典
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
        
        # 获取所有列名
        columns = df.columns.tolist()
        
        # 转换为字典列表
        data = df.to_dict('records')
        
        return {
            "columns": columns,
            "data": data
        }
        
    except Exception as e:
        raise ValueError(f"文件读取失败: {str(e)}") 