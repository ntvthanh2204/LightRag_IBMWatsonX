# LightRAG Token và Embedding Monitoring Guide

## Tổng quan

Hướng dẫn này mô tả cách cấu hình và sử dụng TokenTracker và EmbeddingTracker để monitor chi tiết việc sử dụng API của LightRAG với WatsonX và các embedding providers khác.

## Cài đặt

```bash
pip install lightrag python-dotenv httpx tenacity openai
```

## Cấu hình Environment Variables

Tạo file `.env` trong project directory:

```env
# WatsonX Configuration
WATSONX_API_KEY=your_watsonx_api_key
WATSONX_PROJECT_ID=your_project_id
WATSONX_MODEL_ID=openai/gpt-oss-120b
WATSONX_BASE_URL=https://us-south.ml.cloud.ibm.com/ml/v1

# Embedding Configuration (OpenAI example)
OPENAI_API_KEY=your_openai_api_key

# Optional: Logging
VERBOSE=true
LOG_LEVEL=DEBUG
```

## Cấu hình TokenTracker cho WatsonX

### 1. Import Required Modules

```python
from lightrag.utils import TokenTracker, EmbeddingTracker, setup_logger
from lightrag.llm.watsonx import watsonx_llm_acomplete
from lightrag import LightRAG, QueryParam
```

### 2. Tạo Token Tracker

```python
# Khởi tạo trackers
token_tracker = TokenTracker()
embedding_tracker = EmbeddingTracker()
```

### 3. Wrapper Function cho WatsonX

```python
async def watsonx_llm_func_with_tracking(
    prompt: str, 
    system_prompt: str = None, 
    history_messages: list = None, 
    **kwargs
) -> str:
    """WatsonX LLM function với token tracking"""
    # Thêm token_tracker vào kwargs
    kwargs["token_tracker"] = token_tracker
    
    return await watsonx_llm_acomplete(
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )
```

## Cấu hình EmbeddingTracker

### 1. Embedding Function với Tracking

```python
async def embedding_func_with_tracking(texts: list[str]) -> np.ndarray:
    """Embedding function với detailed tracking"""
    import openai
    
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    # Tính toán metrics
    total_chars = sum(len(text) for text in texts)
    estimated_tokens = total_chars // 4  # 4 chars ≈ 1 token
    
    try:
        # Gọi embedding API
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        
        # Extract embeddings
        embeddings = np.array([item.embedding for item in response.data])
        
        # Track usage
        actual_tokens = response.usage.total_tokens if hasattr(response, 'usage') else estimated_tokens
        embedding_dim = len(embeddings[0]) if len(embeddings) > 0 else 1536
        
        embedding_tracker.add_usage(
            texts_count=len(texts),
            total_chars=total_chars,
            tokens_used=actual_tokens,
            embedding_dim=embedding_dim
        )
        
        return embeddings.astype(np.float32)
        
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        # Fallback với tracking
        embedding_tracker.add_usage(
            texts_count=len(texts),
            total_chars=total_chars,
            tokens_used=estimated_tokens,
            embedding_dim=1536
        )
        return np.random.rand(len(texts), 1536).astype(np.float32)
```

## Cấu hình LightRAG với Comprehensive Tracking

```python
async def initialize_rag_with_tracking():
    """Khởi tạo LightRAG với comprehensive tracking"""
    rag = LightRAG(
        working_dir="./rag_storage",
        # LLM configuration với tracking
        llm_model_func=watsonx_llm_func_with_tracking,
        # Embedding configuration với tracking
        embedding_func=EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=embedding_func_with_tracking,
        ),
        # Performance settings
        entity_extract_max_gleaning=1,
        # Caching để tối ưu usage
        enable_llm_cache=True,
        enable_llm_cache_for_entity_extract=True,
        embedding_cache_config={
            "enabled": True, 
            "similarity_threshold": 0.90
        },
    )

    await rag.initialize_storages()
    return rag
```

## Sử dụng Context Managers

### Method 1: Context Manager (Recommended)

```python
# Tự động tracking và in statistics khi kết thúc
with token_tracker:
    with embedding_tracker:
        # Insert documents
        await rag.ainsert(document_text)
        
        # Query với different modes
        result = await rag.aquery("What is AI?", param=QueryParam(mode="global"))
        
# Statistics tự động được in ra khi thoát context
```

### Method 2: Manual Tracking

```python
# Reset counters
token_tracker.reset()
embedding_tracker.reset()

# Perform operations
await rag.ainsert(document)
result = await rag.aquery("question")

# Manually get statistics
print(token_tracker)
print(embedding_tracker)
```

## Detailed Usage Statistics

### Token Usage Statistics

```python
token_stats = token_tracker.get_usage()
print(f"LLM Calls: {token_stats['call_count']}")
print(f"Prompt Tokens: {token_stats['prompt_tokens']:,}")
print(f"Completion Tokens: {token_stats['completion_tokens']:,}")
print(f"Total Tokens: {token_stats['total_tokens']:,}")

# Average calculations
if token_stats['call_count'] > 0:
    avg_prompt = token_stats['prompt_tokens'] / token_stats['call_count']
    avg_completion = token_stats['completion_tokens'] / token_stats['call_count']
    print(f"Avg Prompt/Call: {avg_prompt:.1f}")
    print(f"Avg Completion/Call: {avg_completion:.1f}")
```

### Embedding Usage Statistics

```python
embed_stats = embedding_tracker.get_usage()
print(f"Embedding Calls: {embed_stats['embedding_calls']}")
print(f"Total Texts: {embed_stats['total_texts']:,}")
print(f"Total Characters: {embed_stats['total_characters']:,}")
print(f"Total Tokens: {embed_stats['total_tokens']:,}")
print(f"Dimensions: {embed_stats['embedding_dimensions']}")
print(f"Avg Chars/Text: {embed_stats['avg_chars_per_text']:.1f}")
print(f"Avg Tokens/Text: {embed_stats['avg_tokens_per_text']:.1f}")
```

## Cost Estimation

```python
def calculate_estimated_costs(token_stats, embed_stats):
    """Calculate estimated costs based on provider pricing"""
    
    # WatsonX pricing (example rates)
    watsonx_input_rate = 0.001 / 1000  # $0.001 per 1K tokens
    watsonx_output_rate = 0.002 / 1000  # $0.002 per 1K tokens
    
    # OpenAI embedding pricing
    embedding_rate = 0.00002 / 1000  # $0.00002 per 1K tokens
    
    # Calculate costs
    llm_input_cost = token_stats['prompt_tokens'] * watsonx_input_rate
    llm_output_cost = token_stats['completion_tokens'] * watsonx_output_rate
    embedding_cost = embed_stats['total_tokens'] * embedding_rate
    
    total_cost = llm_input_cost + llm_output_cost + embedding_cost
    
    return {
        "llm_input_cost": llm_input_cost,
        "llm_output_cost": llm_output_cost,
        "embedding_cost": embedding_cost,
        "total_cost": total_cost
    }
```

## Advanced Monitoring Features

### 1. Per-Query Tracking

```python
async def track_per_query(rag, query, mode="global"):
    """Track usage cho một query cụ thể"""
    query_token_tracker = TokenTracker()
    query_embedding_tracker = EmbeddingTracker()
    
    # Temporarily replace global trackers
    global token_tracker, embedding_tracker
    original_token = token_tracker
    original_embed = embedding_tracker
    
    token_tracker = query_token_tracker
    embedding_tracker = query_embedding_tracker
    
    try:
        result = await rag.aquery(query, param=QueryParam(mode=mode))
        
        # Get query-specific stats
        query_stats = {
            "query": query,
            "mode": mode,
            "tokens": query_token_tracker.get_usage(),
            "embeddings": query_embedding_tracker.get_usage(),
            "result_length": len(result)
        }
        
        return result, query_stats
        
    finally:
        # Restore global trackers
        token_tracker = original_token
        embedding_tracker = original_embed
```

### 2. Export Usage Reports

```python
def export_usage_report(token_tracker, embedding_tracker, output_file="usage_report.json"):
    """Export detailed usage report"""
    import json
    from datetime import datetime
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_config": {
            "watsonx_model": os.getenv("WATSONX_MODEL_ID"),
            "embedding_model": "text-embedding-3-small",
        },
        "token_usage": token_tracker.get_usage(),
        "embedding_usage": embedding_tracker.get_usage(),
        "estimated_costs": calculate_estimated_costs(
            token_tracker.get_usage(), 
            embedding_tracker.get_usage()
        )
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Usage report exported to: {output_file}")
```

### 3. Real-time Monitoring

```python
def setup_realtime_monitoring():
    """Setup real-time monitoring với logging"""
    import logging
    
    # Setup custom logger cho monitoring
    monitor_logger = logging.getLogger("token_monitor")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - MONITOR - %(message)s'
    )
    handler.setFormatter(formatter)
    monitor_logger.addHandler(handler)
    monitor_logger.setLevel(logging.INFO)
    
    # Hook vào TokenTracker
    original_add_usage = token_tracker.add_usage
    
    def monitored_add_usage(token_counts):
        result = original_add_usage(token_counts)
        monitor_logger.info(
            f"LLM Call - Prompt: {token_counts.get('prompt_tokens', 0)}, "
            f"Completion: {token_counts.get('completion_tokens', 0)}, "
            f"Total: {token_counts.get('total_tokens', 0)}"
        )
        return result
    
    token_tracker.add_usage = monitored_add_usage
```

## Best Practices

### 1. Caching để tối ưu costs

```python
# Enable caching để giảm API calls
rag_config = {
    "enable_llm_cache": True,
    "enable_llm_cache_for_entity_extract": True,
    "embedding_cache_config": {
        "enabled": True,
        "similarity_threshold": 0.90  # Cache nếu similarity > 90%
    }
}
```

### 2. Batch Processing

```python
# Process documents in batches
async def process_documents_in_batches(rag, documents, batch_size=5):
    """Process documents in batches để tối ưu API usage"""
    total_docs = len(documents)
    
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_docs-1)//batch_size + 1}")
        
        for doc in batch:
            await rag.ainsert(doc)
        
        # Print intermediate stats
        print(f"Current usage: {token_tracker}")
        print(f"Current embeddings: {embedding_tracker}")
```

### 3. Error Handling với Tracking

```python
async def safe_query_with_tracking(rag, query, max_retries=3):
    """Safe query với tracking và retry logic"""
    for attempt in range(max_retries):
        try:
            with token_tracker:
                with embedding_tracker:
                    result = await rag.aquery(query)
                    return result
                    
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            
            # Wait before retry
            import asyncio
            await asyncio.sleep(2 ** attempt)
```

## Running the Demo

```bash
# Chạy demo với comprehensive tracking
python examples/lightrag_watsonx_track_token_demo.py
```

Demo sẽ:
1. ✅ Insert sample document với tracking
2. 🔍 Test 4 query modes (naive, local, global, hybrid)
3. 📊 In detailed statistics cho mỗi operation
4. 💰 Tính estimated costs
5. 📋 Export usage report

## Troubleshooting

### Common Issues

1. **WatsonX API Authentication Error**
   ```bash
   ❌ Authentication failed: Invalid API key
   ```
   - Kiểm tra `WATSONX_API_KEY` và `WATSONX_PROJECT_ID`

2. **Embedding API Error**
   ```bash
   ❌ Embedding error: Unauthorized
   ```
   - Kiểm tra `OPENAI_API_KEY` hoặc embedding provider credentials

3. **Token Tracking Not Working**
   - Đảm bảo pass `token_tracker` trong kwargs của LLM function
   - Kiểm tra WatsonX API response có chứa usage information

### Debug Mode

```python
# Enable verbose logging
import os
os.environ["VERBOSE"] = "true"
os.environ["LOG_LEVEL"] = "DEBUG"

# Enable detailed tracking logs
setup_logger("lightrag", level="DEBUG")
```

## Kết luận

TokenTracker và EmbeddingTracker cung cấp monitoring toàn diện cho LightRAG applications, giúp:

- 📊 Track detailed usage statistics
- 💰 Estimate API costs accurately  
- 🔍 Monitor per-query performance
- 📋 Export comprehensive reports
- ⚡ Optimize caching strategies
- 🛡️ Implement error handling với tracking

Sử dụng các công cụ này để có visibility tốt nhất về API usage và costs trong production environments.