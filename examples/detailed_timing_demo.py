"""
LightRAG Detailed Timing Demo

Demo này cho phép bạn xem CHÍNH XÁC thời gian mỗi lần gọi LLM và embedding,
bao gồm phân tích chi tiết performance và export detailed reports.

Cài đặt:
pip install lightrag python-dotenv httpx tenacity openai
"""

import os
import asyncio
import time
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

from lightrag.utils import (
    EmbeddingFunc, 
    TokenTracker, 
    EmbeddingTracker, 
    setup_logger
)
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.watsonx import watsonx_llm_acomplete

# Setup
setup_logger("lightrag", level="INFO")
load_dotenv()

# Configuration
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID", "openai/gpt-oss-120b")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WORKING_DIR = "./timing_demo"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Global trackers với detailed logging enabled
token_tracker = TokenTracker(log_each_call=True)
embedding_tracker = EmbeddingTracker(log_each_call=True)


class TimingDecorator:
    """Decorator để measure timing của bất kỳ function nào"""
    
    def __init__(self, operation_name="operation"):
        self.operation_name = operation_name
        
    def __call__(self, func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                print(f"⏱️  Starting {self.operation_name}...")
                
                try:
                    result = await func(*args, **kwargs)
                    end_time = time.time()
                    duration = end_time - start_time
                    print(f"✅ {self.operation_name} completed in {duration:.2f}s")
                    return result
                except Exception as e:
                    end_time = time.time()
                    duration = end_time - start_time
                    print(f"❌ {self.operation_name} failed after {duration:.2f}s: {e}")
                    raise
                    
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                print(f"⏱️  Starting {self.operation_name}...")
                
                try:
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    duration = end_time - start_time
                    print(f"✅ {self.operation_name} completed in {duration:.2f}s")
                    return result
                except Exception as e:
                    end_time = time.time()
                    duration = end_time - start_time
                    print(f"❌ {self.operation_name} failed after {duration:.2f}s: {e}")
                    raise
                    
            return sync_wrapper


@TimingDecorator("WatsonX LLM Call")
async def watsonx_llm_func_with_detailed_timing(
    prompt: str, 
    system_prompt: str = None, 
    history_messages: list = None,
    operation_type: str = None,
    **kwargs
) -> str:
    """WatsonX LLM function với chi tiết timing tracking"""
    # Thêm metadata cho tracking
    kwargs["token_tracker"] = token_tracker
    kwargs["operation_type"] = operation_type
    
    return await watsonx_llm_acomplete(
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )


@TimingDecorator("Embedding API Call") 
async def embedding_func_with_detailed_timing(texts: list[str]) -> np.ndarray:
    """Embedding function với detailed timing tracking"""
    import openai
    
    start_time = time.time()
    
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    # Calculate metrics
    total_chars = sum(len(text) for text in texts)
    batch_size = len(texts)
    
    try:
        # Call embedding API
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        
        end_time = time.time()
        call_time = end_time - start_time
        
        # Extract embeddings
        embeddings = np.array([item.embedding for item in response.data])
        
        # Get actual usage if available
        actual_tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0
        embedding_dim = len(embeddings[0]) if len(embeddings) > 0 else 1536
        
        # Track với detailed timing
        embedding_tracker.add_usage(
            texts_count=batch_size,
            total_chars=total_chars,
            tokens_used=actual_tokens,
            embedding_dim=embedding_dim,
            call_time=call_time,
            model_name="text-embedding-3-small",
            operation_type="text_embedding",
            batch_size=batch_size
        )
        
        return embeddings.astype(np.float32)
        
    except Exception as e:
        end_time = time.time()
        call_time = end_time - start_time
        
        # Track failed call với estimated values
        estimated_tokens = total_chars // 4
        embedding_tracker.add_usage(
            texts_count=batch_size,
            total_chars=total_chars,
            tokens_used=estimated_tokens,
            embedding_dim=1536,
            call_time=call_time,
            model_name="text-embedding-3-small",
            operation_type="text_embedding_failed",
            batch_size=batch_size
        )
        
        print(f"❌ Embedding API error: {e}")
        # Return dummy embeddings
        return np.random.rand(len(texts), 1536).astype(np.float32)


@TimingDecorator("LightRAG Initialization")
async def initialize_rag_with_detailed_tracking():
    """Initialize LightRAG với comprehensive timing tracking"""
    rag = LightRAG(
        working_dir=WORKING_DIR,
        # LLM với timing
        llm_model_func=watsonx_llm_func_with_detailed_timing,
        # Embedding với timing  
        embedding_func=EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=embedding_func_with_detailed_timing,
        ),
        # Optimization settings
        entity_extract_max_gleaning=1,
        enable_llm_cache=True,
        enable_llm_cache_for_entity_extract=True,
        embedding_cache_config={
            "enabled": True,
            "similarity_threshold": 0.90
        },
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


def print_real_time_stats():
    """In real-time statistics"""
    print("\n" + "="*70)
    print("⏱️  REAL-TIME PERFORMANCE STATISTICS")
    print("="*70)
    
    # LLM Stats
    print("🔤 LLM PERFORMANCE:")
    if token_tracker.call_count > 0:
        usage = token_tracker.get_usage()
        timing_stats = token_tracker.get_timing_stats()
        
        print(f"   📊 Calls: {usage['call_count']}")
        print(f"   ⏱️  Total Time: {usage['total_time']:.2f}s")
        print(f"   📈 Avg Time/Call: {usage['avg_time_per_call']:.2f}s")
        print(f"   ⚡ Speed: {usage['overall_tokens_per_second']:.1f} tokens/s")
        print(f"   📏 Range: {usage['min_time']:.2f}s - {usage['max_time']:.2f}s")
        
        if timing_stats:
            print(f"   📊 Median Time: {timing_stats['median_time']:.2f}s")
            print(f"   📐 Std Dev: {timing_stats['std_dev']:.2f}s")
    else:
        print("   📭 No LLM calls recorded yet")
    
    print()
    
    # Embedding Stats
    print("🔢 EMBEDDING PERFORMANCE:")
    if embedding_tracker.embedding_calls > 0:
        usage = embedding_tracker.get_usage()
        timing_stats = embedding_tracker.get_timing_stats()
        
        print(f"   📊 Calls: {usage['embedding_calls']}")
        print(f"   ⏱️  Total Time: {usage['total_time']:.2f}s")
        print(f"   📈 Avg Time/Call: {usage['avg_time_per_call']:.2f}s")
        print(f"   ⚡ Speed: {usage['overall_texts_per_second']:.1f} texts/s, {usage['overall_tokens_per_second']:.1f} tokens/s")
        print(f"   📏 Range: {usage['min_time']:.2f}s - {usage['max_time']:.2f}s")
        
        if timing_stats:
            print(f"   📊 Median Time: {timing_stats['median_time']:.2f}s")
            print(f"   📐 Std Dev: {timing_stats['std_dev']:.2f}s")
    else:
        print("   📭 No embedding calls recorded yet")
    
    print("="*70)


def show_detailed_call_history():
    """Hiển thị detailed call history"""
    print("\n" + "="*70)
    print("📋 DETAILED CALL HISTORY")
    print("="*70)
    
    # LLM Call History
    print("\n🔤 LLM CALL HISTORY:")
    llm_history = token_tracker.get_call_history()
    if llm_history:
        for call in llm_history[-5:]:  # Show last 5 calls
            print(f"   #{call['call_number']} | "
                  f"{call['timestamp'][:19]} | "
                  f"Model: {call['model_name']} | "
                  f"Operation: {call['operation_type']} | "
                  f"Time: {call['call_time']:.2f}s | "
                  f"Tokens: {call['total_tokens']} | "
                  f"Speed: {call['tokens_per_second']:.1f}/s")
    else:
        print("   📭 No LLM call history")
    
    # Embedding Call History  
    print("\n🔢 EMBEDDING CALL HISTORY:")
    embed_history = embedding_tracker.get_call_history()
    if embed_history:
        for call in embed_history[-5:]:  # Show last 5 calls
            print(f"   #{call['call_number']} | "
                  f"{call['timestamp'][:19]} | "
                  f"Model: {call['model_name']} | "
                  f"Operation: {call['operation_type']} | "
                  f"Time: {call['call_time']:.2f}s | "
                  f"Texts: {call['texts_count']} | "
                  f"Speed: {call['texts_per_second']:.1f} texts/s")
    else:
        print("   📭 No embedding call history")
    
    print("="*70)


@TimingDecorator("Document Insert")
async def insert_document_with_timing(rag, text, doc_name="Sample Document"):
    """Insert document với detailed timing tracking"""
    print(f"\n📝 Inserting document: {doc_name}")
    print(f"📏 Document length: {len(text)} characters")
    
    await rag.ainsert(text)
    print_real_time_stats()


@TimingDecorator("Query Execution")
async def query_with_timing(rag, query, mode="global"):
    """Execute query với detailed timing tracking"""
    print(f"\n🔍 Executing query: {query}")
    print(f"🎯 Mode: {mode}")
    
    result = await rag.aquery(query, param=QueryParam(mode=mode))
    
    print(f"📝 Result length: {len(result)} characters")
    print(f"📄 Result preview: {result[:100]}...")
    
    print_real_time_stats()
    return result


async def run_comprehensive_timing_demo():
    """Chạy demo comprehensive với detailed timing"""
    print("🚀 LightRAG Detailed Timing Analysis Demo")
    print("="*70)
    
    # Check credentials
    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        print("❌ WatsonX credentials missing!")
        return
    
    if not OPENAI_API_KEY:
        print("⚠️  OpenAI API key missing - using dummy embeddings")
    
    # Initialize RAG
    print("\n📊 Initializing LightRAG with detailed timing...")
    rag = await initialize_rag_with_detailed_tracking()
    
    # Sample documents với different sizes
    documents = [
        {
            "name": "Small Document", 
            "text": "Artificial Intelligence is transforming business operations across industries."
        },
        {
            "name": "Medium Document",
            "text": """
            Artificial Intelligence (AI) is revolutionizing the business landscape across multiple industries. 
            Companies are increasingly adopting machine learning algorithms to automate processes, enhance 
            customer experiences, and drive innovation. Key applications include natural language processing 
            for chatbots, computer vision for quality control, and predictive analytics for forecasting. 
            However, successful AI implementation requires careful planning, data quality management, and 
            ethical considerations. Organizations must invest in talent development and infrastructure to 
            fully realize AI's potential while addressing challenges such as data privacy and algorithmic bias.
            """
        }
    ]
    
    # Insert documents với timing
    for doc in documents:
        await insert_document_with_timing(rag, doc["text"], doc["name"])
        print(f"\n⏸️  Pausing 2s between documents...")
        await asyncio.sleep(2)
    
    # Test different query modes với timing
    queries = [
        "What is AI?",
        "How does AI transform business?", 
        "What are the challenges of AI implementation?"
    ]
    
    query_modes = ["naive", "local", "global", "hybrid"]
    
    for i, query in enumerate(queries, 1):
        print(f"\n\n🔍 QUERY SET {i}: {query}")
        print("-" * 50)
        
        for mode in query_modes:
            await query_with_timing(rag, query, mode)
            print(f"\n⏸️  Pausing 1s between modes...")
            await asyncio.sleep(1)
        
        # Show call history after each query set
        show_detailed_call_history()
    
    # Final comprehensive analysis
    print("\n\n🎯 FINAL COMPREHENSIVE ANALYSIS")
    print("="*70)
    
    # Export detailed reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    llm_report = token_tracker.export_call_history(
        f"{WORKING_DIR}/llm_timing_report_{timestamp}.json"
    )
    
    embed_report = embedding_tracker.export_call_history(
        f"{WORKING_DIR}/embedding_timing_report_{timestamp}.json"
    )
    
    # Performance analysis
    print("\n📊 PERFORMANCE ANALYSIS:")
    
    # LLM Performance
    llm_timing = token_tracker.get_timing_stats()
    if llm_timing:
        print(f"\n🔤 LLM Performance Metrics:")
        print(f"   🎯 Total Calls: {llm_timing['total_calls']}")
        print(f"   ⏱️  Total Time: {llm_timing['total_time']:.2f}s")
        print(f"   📈 Average: {llm_timing['avg_time']:.2f}s")
        print(f"   📊 Median: {llm_timing['median_time']:.2f}s")
        print(f"   📏 Range: {llm_timing['min_time']:.2f}s - {llm_timing['max_time']:.2f}s")
        print(f"   📐 Std Dev: {llm_timing['std_dev']:.2f}s")
        
        # Performance recommendations
        if llm_timing['std_dev'] > 2.0:
            print("   ⚠️  High variance in response times - consider optimization")
        if llm_timing['avg_time'] > 5.0:
            print("   ⚠️  Slow average response time - check network/model")
        if llm_timing['avg_time'] < 1.0:
            print("   ✅ Good average response time")
    
    # Embedding Performance
    embed_timing = embedding_tracker.get_timing_stats()
    if embed_timing:
        print(f"\n🔢 Embedding Performance Metrics:")
        print(f"   🎯 Total Calls: {embed_timing['total_calls']}")
        print(f"   ⏱️  Total Time: {embed_timing['total_time']:.2f}s")
        print(f"   📈 Average: {embed_timing['avg_time']:.2f}s")
        print(f"   📊 Median: {embed_timing['median_time']:.2f}s")
        print(f"   📏 Range: {embed_timing['min_time']:.2f}s - {embed_timing['max_time']:.2f}s")
        print(f"   📐 Std Dev: {embed_timing['std_dev']:.2f}s")
        
        # Performance recommendations
        embed_perf = embedding_tracker.get_performance_stats()
        if embed_perf and "texts_per_second" in embed_perf:
            avg_speed = embed_perf["texts_per_second"]["avg"]
            print(f"   ⚡ Average Speed: {avg_speed:.1f} texts/second")
            
            if avg_speed < 10:
                print("   ⚠️  Slow embedding speed - consider batching optimization")
            elif avg_speed > 50:
                print("   ✅ Good embedding throughput")
    
    # Cost analysis (example rates)
    total_llm_tokens = token_tracker.get_usage()["total_tokens"]
    total_embed_tokens = embedding_tracker.get_usage()["total_tokens"]
    
    # Estimated costs (example rates)
    llm_cost = total_llm_tokens * 0.002 / 1000  # Example: $0.002/1K tokens
    embed_cost = total_embed_tokens * 0.00002 / 1000  # Example: $0.00002/1K tokens
    total_cost = llm_cost + embed_cost
    
    print(f"\n💰 COST ANALYSIS (Estimated):")
    print(f"   🔤 LLM Cost: ${llm_cost:.6f} ({total_llm_tokens:,} tokens)")
    print(f"   🔢 Embedding Cost: ${embed_cost:.6f} ({total_embed_tokens:,} tokens)")
    print(f"   💸 Total Estimated: ${total_cost:.6f}")
    
    print(f"\n📋 REPORTS EXPORTED:")
    print(f"   📄 LLM Report: {llm_report}")
    print(f"   📄 Embedding Report: {embed_report}")
    
    print(f"\n✅ Demo completed! Check {WORKING_DIR} for detailed reports.")


def main():
    """Main function"""
    try:
        asyncio.run(run_comprehensive_timing_demo())
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 Demo finished!")


if __name__ == "__main__":
    main()