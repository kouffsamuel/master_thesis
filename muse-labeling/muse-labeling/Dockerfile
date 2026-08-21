FROM nginx:alpine
COPY dist/ /usr/share/nginx/html
COPY nginx/nginx.conf /etc/nginx/conf.d/default.conf
COPY nginx/cert.pem /etc/nginx/certs/cert.pem
COPY nginx/key.pem  /etc/nginx/certs/key.pem

# npm run build
# sudo docker build -t afaifai/muse-labeling:latest .
# sudo docker push afaifai/muse-labeling:latest

# sudo docker pull afaifai/muse-labeling:latest
# sudo docker run -p 443:443 afaifai/muse-labeling:latest
# https://localhost