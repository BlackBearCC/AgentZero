FROM node:18-alpine as build-stage

# 设置环境变量以解决 crypto.getRandomValues 问题
ENV NODE_OPTIONS=--openssl-legacy-provider

WORKDIR /app
COPY package*.json ./

# Install dependencies including Element Plus
RUN npm install
RUN npm install element-plus @element-plus/icons-vue

# 复制所有源文件，包括字体
COPY . .

# Add Element Plus import to main.js if not already present
RUN if ! grep -q "element-plus" src/main.js; then \
    echo "import ElementPlus from 'element-plus';" >> src/main.js && \
    echo "import 'element-plus/dist/index.css';" >> src/main.js && \
    echo "app.use(ElementPlus);" >> src/main.js; \
    fi

RUN npm run build

FROM nginx:stable-alpine as production-stage
# 复制构建产物
COPY --from=build-stage /app/dist /usr/share/nginx/html
# 确保字体目录存在并复制字体文件
RUN mkdir -p /usr/share/nginx/html/fonts

COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]