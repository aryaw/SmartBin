FROM node:22-slim AS builder

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:22-slim AS runner
WORKDIR /app
COPY --from=builder /app/.output ./.output
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./
COPY log-wrapper.sh /app/log-wrapper.sh
RUN chmod +x /app/log-wrapper.sh

RUN mkdir -p log

EXPOSE 3000

ENTRYPOINT ["/app/log-wrapper.sh"]
CMD ["node", ".output/server/index.mjs"]
