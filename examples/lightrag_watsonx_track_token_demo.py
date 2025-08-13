"""
LightRAG WatsonX Token Tracking Demo

Ví dụ này minh họa cách sử dụng TokenTracker và EmbeddingTracker để monitor
chi tiết việc sử dụng token của WatsonX API và embedding API.

Cài đặt dependencies:
pip install lightrag python-dotenv httpx tenacity
"""

import os
import asyncio
import numpy as np
import nest_asyncio
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc, TokenTracker, EmbeddingTracker, setup_logger
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.watsonx import watsonx_llm_acomplete

# Thiết lập logging
setup_logger("lightrag", level="DEBUG")

# Apply nest_asyncio để giải quyết vấn đề event loop
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Cấu hình WatsonX
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID", "openai/gpt-oss-120b")

# Cấu hình embedding (sử dụng OpenAI hoặc provider khác)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

WORKING_DIR = "./watsonx_rag_demo"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Khởi tạo trackers
token_tracker = TokenTracker()
embedding_tracker = EmbeddingTracker()


async def watsonx_llm_func_with_tracking(
    prompt: str, 
    system_prompt: str = None, 
    history_messages: list = None, 
    **kwargs
) -> str:
    """
    WatsonX LLM function với token tracking
    """
    # Thêm token_tracker vào kwargs
    kwargs["token_tracker"] = token_tracker
    
    return await watsonx_llm_acomplete(
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )


async def embedding_func_with_tracking(texts: list[str]) -> np.ndarray:
    """
    Embedding function với detailed tracking
    """
    import openai
    
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    # Tính toán metrics
    total_chars = sum(len(text) for text in texts)
    
    # Ước tính tokens (4 chars ≈ 1 token)
    estimated_tokens = total_chars // 4
    
    try:
        # Gọi OpenAI embedding API
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
        
        print(f"📊 Embedding Call - Texts: {len(texts)}, Chars: {total_chars}, "
              f"Tokens: {actual_tokens}, Dim: {embedding_dim}")
        
        return embeddings.astype(np.float32)
        
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        # Fallback to random embeddings với tracking
        embedding_tracker.add_usage(
            texts_count=len(texts),
            total_chars=total_chars,
            tokens_used=estimated_tokens,
            embedding_dim=1536
        )
        return np.random.rand(len(texts), 1536).astype(np.float32)


async def initialize_rag_with_tracking():
    """
    Khởi tạo LightRAG với comprehensive tracking
    """
    rag = LightRAG(
        working_dir=WORKING_DIR,
        # LLM configuration
        llm_model_func=watsonx_llm_func_with_tracking,
        # Embedding configuration
        embedding_func=EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=embedding_func_with_tracking,
        ),
        # Performance settings
        entity_extract_max_gleaning=1,
        # Caching settings
        enable_llm_cache=True,
        enable_llm_cache_for_entity_extract=True,
        embedding_cache_config={
            "enabled": True, 
            "similarity_threshold": 0.90
        },
        # Advanced settings
        graph_storage_config={
            "provider": "json",
        }
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def print_detailed_usage_stats():
    """
    In thống kê chi tiết về token và embedding usage
    """
    print("\n" + "="*80)
    print("📈 DETAILED USAGE STATISTICS")
    print("="*80)
    
    # Token Usage Statistics
    token_stats = token_tracker.get_usage()
    print(f"🔤 LLM TOKEN USAGE:")
    print(f"   • API Calls: {token_stats['call_count']}")
    print(f"   • Prompt Tokens: {token_stats['prompt_tokens']:,}")
    print(f"   • Completion Tokens: {token_stats['completion_tokens']:,}")
    print(f"   • Total Tokens: {token_stats['total_tokens']:,}")
    
    if token_stats['call_count'] > 0:
        avg_prompt = token_stats['prompt_tokens'] / token_stats['call_count']
        avg_completion = token_stats['completion_tokens'] / token_stats['call_count']
        print(f"   • Avg Prompt/Call: {avg_prompt:.1f}")
        print(f"   • Avg Completion/Call: {avg_completion:.1f}")
    
    print()
    
    # Embedding Usage Statistics
    embed_stats = embedding_tracker.get_usage()
    print(f"🔢 EMBEDDING USAGE:")
    print(f"   • API Calls: {embed_stats['embedding_calls']}")
    print(f"   • Total Texts: {embed_stats['total_texts']:,}")
    print(f"   • Total Characters: {embed_stats['total_characters']:,}")
    print(f"   • Total Tokens: {embed_stats['total_tokens']:,}")
    print(f"   • Embedding Dimensions: {embed_stats['embedding_dimensions']}")
    print(f"   • Avg Chars/Text: {embed_stats['avg_chars_per_text']:.1f}")
    print(f"   • Avg Tokens/Text: {embed_stats['avg_tokens_per_text']:.1f}")
    
    if embed_stats['embedding_calls'] > 0:
        avg_texts_per_call = embed_stats['total_texts'] / embed_stats['embedding_calls']
        print(f"   • Avg Texts/Call: {avg_texts_per_call:.1f}")
    
    # Cost Estimation (ví dụ với giá OpenAI)
    print()
    print("💰 ESTIMATED COSTS (OpenAI Pricing):")
    
    # LLM costs (ví dụ: GPT-3.5-turbo)
    prompt_cost = token_stats['prompt_tokens'] * 0.001 / 1000  # $0.001/1K tokens
    completion_cost = token_stats['completion_tokens'] * 0.002 / 1000  # $0.002/1K tokens
    llm_total_cost = prompt_cost + completion_cost
    
    # Embedding costs (text-embedding-3-small)
    embed_cost = embed_stats['total_tokens'] * 0.00002 / 1000  # $0.00002/1K tokens
    
    total_cost = llm_total_cost + embed_cost
    
    print(f"   • LLM Costs: ${llm_total_cost:.6f}")
    print(f"     - Prompt: ${prompt_cost:.6f}")
    print(f"     - Completion: ${completion_cost:.6f}")
    print(f"   • Embedding Costs: ${embed_cost:.6f}")
    print(f"   • TOTAL ESTIMATED: ${total_cost:.6f}")
    
    print("="*80)


async def run_comprehensive_demo():
    """
    Chạy demo toàn diện với chi tiết monitoring
    """
    print("🚀 Starting LightRAG WatsonX Token Tracking Demo")
    print(f"📁 Working Directory: {WORKING_DIR}")
    
    # Kiểm tra API keys
    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        print("❌ WatsonX API credentials not found!")
        print("Please set WATSONX_API_KEY and WATSONX_PROJECT_ID environment variables")
        return
    
    if not OPENAI_API_KEY:
        print("⚠️  OpenAI API key not found! Using dummy embeddings.")
    
    # Khởi tạo RAG với tracking
    print("\n📊 Initializing LightRAG with comprehensive tracking...")
    rag = await initialize_rag_with_tracking()
    
    # Sample document
    sample_text = """
    Artificial Intelligence (AI) is transforming the business landscape across industries. 
    Companies are leveraging machine learning algorithms to automate processes, enhance 
    customer experiences, and drive innovation. Key applications include natural language 
    processing for chatbots, computer vision for quality control, and predictive analytics 
    for forecasting. However, successful AI implementation requires careful planning, 
    data quality management, and ethical considerations. Organizations must invest in 
    talent development and infrastructure to fully realize AI's potential.
    """
    
    print("\n📝 Inserting sample document...")
    print(f"Document length: {len(sample_text)} characters")
    
    # Insert document với tracking
    with token_tracker:
        with embedding_tracker:
            await rag.ainsert(sample_text)
    
    print("\n✅ Document inserted successfully!")
    print_detailed_usage_stats()
    
    # Test queries với different modes
    queries = [
        "What are the key applications of AI in business?",
        "How can companies implement AI successfully?",
        "What challenges do organizations face with AI adoption?"
    ]
    
    query_modes = ["naive", "local", "global", "hybrid"]
    
    for i, query in enumerate(queries, 1):
        print(f"\n🔍 QUERY {i}: {query}")
        print("-" * 60)
        
        for mode in query_modes:
            print(f"\n📊 Testing {mode.upper()} mode...")
            
            # Reset counters for this query
            query_token_tracker = TokenTracker()
            query_embedding_tracker = EmbeddingTracker()
            
            # Temporarily replace global trackers
            global token_tracker, embedding_tracker
            original_token_tracker = token_tracker
            original_embedding_tracker = embedding_tracker
            
            token_tracker = query_token_tracker
            embedding_tracker = query_embedding_tracker
            
            try:
                with token_tracker:
                    with embedding_tracker:
                        result = await rag.aquery(
                            query, 
                            param=QueryParam(mode=mode, only_need_context=False)
                        )
                
                # In kết quả ngắn gọn
                print(f"✅ {mode.capitalize()} result: {result[:100]}...")
                
                # In stats cho query này
                query_token_stats = query_token_tracker.get_usage()
                query_embed_stats = query_embedding_tracker.get_usage()
                
                print(f"   📊 Tokens: {query_token_stats['total_tokens']} | "
                      f"Calls: {query_token_stats['call_count']} | "
                      f"Embeddings: {query_embed_stats['embedding_calls']}")
                
            except Exception as e:
                print(f"❌ Error in {mode} mode: {e}")
            
            finally:
                # Khôi phục global trackers và cộng dồn stats
                original_token_tracker.prompt_tokens += query_token_tracker.prompt_tokens
                original_token_tracker.completion_tokens += query_token_tracker.completion_tokens
                original_token_tracker.total_tokens += query_token_tracker.total_tokens
                original_token_tracker.call_count += query_token_tracker.call_count
                
                original_embedding_tracker.embedding_calls += query_embedding_tracker.embedding_calls
                original_embedding_tracker.total_texts += query_embedding_tracker.total_texts
                original_embedding_tracker.total_characters += query_embedding_tracker.total_characters
                original_embedding_tracker.total_tokens += query_embedding_tracker.total_tokens
                if query_embedding_tracker.embedding_dimensions > 0:
                    original_embedding_tracker.embedding_dimensions = query_embedding_tracker.embedding_dimensions
                
                token_tracker = original_token_tracker
                embedding_tracker = original_embedding_tracker
    
    # Final comprehensive statistics
    print("\n🎯 FINAL COMPREHENSIVE STATISTICS")
    print_detailed_usage_stats()
    
    # Export detailed report
    export_usage_report()


def export_usage_report():
    """
    Export usage report to file
    """
    import json
    from datetime import datetime
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_config": {
            "watsonx_model": WATSONX_MODEL_ID,
            "embedding_model": "text-embedding-3-small",
            "working_directory": WORKING_DIR
        },
        "token_usage": token_tracker.get_usage(),
        "embedding_usage": embedding_tracker.get_usage(),
    }
    
    report_file = os.path.join(WORKING_DIR, "usage_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Usage report exported to: {report_file}")


def main():
    """
    Main function để chạy demo
    """
    try:
        asyncio.run(run_comprehensive_demo())
    except KeyboardInterrupt:
        print("\n⏹️  Demo stopped by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 Demo completed!")


if __name__ == "__main__":
    main()