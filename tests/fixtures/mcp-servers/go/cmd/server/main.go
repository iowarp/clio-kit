package main

import (
	"context"
	"log"

	"example.com/example-go/mathutil"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

type Input struct {
	A float64 `json:"a" jsonschema:"first factor"`
	B float64 `json:"b" jsonschema:"second factor"`
}

type Output struct {
	Product float64 `json:"product"`
}

func multiply(_ context.Context, _ *mcp.CallToolRequest, input Input) (*mcp.CallToolResult, Output, error) {
	return nil, Output{Product: mathutil.Multiply(input.A, input.B)}, nil
}

func main() {
	server := mcp.NewServer(&mcp.Implementation{Name: "example-go", Version: "1.0.0"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "multiply", Description: "Multiply two finite numbers"}, multiply)
	if err := server.Run(context.Background(), &mcp.StdioTransport{}); err != nil {
		log.Fatal(err)
	}
}
