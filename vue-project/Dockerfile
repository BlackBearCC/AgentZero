FROM node:18-alpine as build-stage

# 设置环境变量以解决 crypto.getRandomValues 问题
ENV NODE_OPTIONS=--openssl-legacy-provider

WORKDIR /app
COPY package*.json ./
RUN npm install

# 复制所有源文件，包括字体
COPY . .

RUN npm run build

FROM nginx:stable-alpine as production-stage
# 复制构建产物
COPY --from=build-stage /app/dist /usr/share/nginx/html
# 确保字体目录存在并复制字体文件
RUN mkdir -p /usr/share/nginx/html/fonts

COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"] 