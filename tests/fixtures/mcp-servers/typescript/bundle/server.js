import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
const server = new McpServer({ name: "example-typescript", version: "1.0.0" });
server.registerTool("multiply", { description: "Multiply two numbers", inputSchema: { a: z.number(), b: z.number() } }, async ({ a, b }) => ({ content: [{ type: "text", text: JSON.stringify({ product: a * b }) }] }));
await server.connect(new StdioServerTransport());
