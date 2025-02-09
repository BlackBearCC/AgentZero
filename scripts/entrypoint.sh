#!/bin/bash

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ASCII 艺术
echo -e "${BLUE}"
cat << "EOF"
    _                    _   ______          
   / \   __ _  ___ _ __ | |_|__  / ___ _ __ 
  / _ \ / _` |/ _ \ '_ \| __| / / / _ \ '__|
 / ___ \ (_| |  __/ | | | |_ / /_|  __/ |   
/_/   \_\__, |\___|_| |_|\__/____|\___||_|   
        |___/                                
EOF
echo -e "${NC}"

echo -e "${YELLOW}正在启动 AgentZero 服务...${NC}\n"

# 进度条函数
progress() {
    local duration=$1
    local width=50
    local progress=0
    local step=$((100/$duration))
    
    while [ $progress -le 100 ]; do
        local count=$(($progress * $width / 100))
        local spaces=$((width - count))
        
        printf "\r[${GREEN}"
        printf "%-${count}s" | tr ' ' '='
        printf "${NC}"
        printf "%-${spaces}s" | tr ' ' ' '
        printf "] %3d%%" $progress
        
        progress=$(($progress + $step))
        sleep 1
    done
    printf "\n"
}

# 等待 MySQL
echo -e "\n${YELLOW}正在等待 MySQL 就绪...${NC}"
until nc -z -v -w30 mysql 3306 > /dev/null 2>&1; do
    echo -n "."
    sleep 1
done
echo -e "\n${GREEN}MySQL 已就绪!${NC}"
progress 5

# 等待 Redis
echo -e "\n${YELLOW}正在等待 Redis 就绪...${NC}"
until nc -z -v -w30 redis 6379 > /dev/null 2>&1; do
    echo -n "."
    sleep 1
done
echo -e "\n${GREEN}Redis 已就绪!${NC}"
progress 5

# 启动应用
echo -e "\n${YELLOW}正在启动应用服务...${NC}"
progress 3
echo -e "\n${GREEN}所有服务已就绪! 开始启动 AgentZero...${NC}\n"

# 启动应用
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000 