package jev

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// DefaultEndpoint is the System One (JEV) endpoint.
const DefaultEndpoint = "https://api.typesafe.ai/v1/systemone"

// DefaultModel is the judge model id used when the profile does not override it.
const DefaultModel = "jev-latest"

// Client calls a System One endpoint. One call makes one attempt; the timeout
// covers sending the request and reading the response body.
type Client struct {
	endpoint string
	apiKey   string
	http     *http.Client
}

// NewClient builds a client. The key is never logged and is redacted from any
// error body the endpoint echoes back.
func NewClient(endpoint, apiKey string, timeout time.Duration) (*Client, error) {
	if endpoint == "" {
		return nil, errors.New("system one endpoint is required")
	}
	if apiKey == "" {
		return nil, errors.New("system one api key is required")
	}
	if strings.ContainsAny(apiKey, "\r\n") {
		return nil, errors.New("system one api key cannot be encoded as an HTTP header")
	}
	if timeout <= 0 {
		timeout = 5 * time.Second
	}
	return &Client{
		endpoint: endpoint,
		apiKey:   apiKey,
		http:     &http.Client{Timeout: timeout},
	}, nil
}

// Call encodes the request, posts it, and decodes the reply into the neutral IR.
func (c *Client) Call(ctx context.Context, request DecisionRequest) (DecisionResponse, error) {
	body, err := EncodeRequest(request)
	if err != nil {
		return DecisionResponse{}, err
	}
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return DecisionResponse{}, err
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set("Authorization", "Bearer "+c.apiKey)

	response, err := c.http.Do(httpRequest)
	if err != nil {
		return DecisionResponse{}, err
	}
	defer response.Body.Close()

	payload, err := io.ReadAll(response.Body)
	if err != nil {
		return DecisionResponse{}, err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return DecisionResponse{}, fmt.Errorf("system one returned %s: %s",
			response.Status, redact(string(payload), c.apiKey))
	}
	return DecodeResponse(payload, request)
}

func redact(text, secret string) string {
	if secret == "" {
		return text
	}
	return strings.ReplaceAll(text, secret, "[REDACTED]")
}
