"""
LightRAG Timing Data Analyzer

Script này phân tích detailed timing data từ TokenTracker và EmbeddingTracker,
tạo charts, statistics và recommendations để optimize performance.

Usage:
python timing_analyzer.py --llm-report llm_timing_report.json --embed-report embedding_timing_report.json
"""

import json
import argparse
import statistics
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class TimingAnalyzer:
    """Analyzer cho timing data từ LightRAG trackers"""
    
    def __init__(self, llm_report_path: str = None, embed_report_path: str = None):
        self.llm_data = None
        self.embed_data = None
        
        if llm_report_path:
            self.load_llm_report(llm_report_path)
        if embed_report_path:
            self.load_embed_report(embed_report_path)
    
    def load_llm_report(self, file_path: str):
        """Load LLM timing report"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.llm_data = json.load(f)
            print(f"✅ Loaded LLM report: {file_path}")
        except Exception as e:
            print(f"❌ Failed to load LLM report: {e}")
    
    def load_embed_report(self, file_path: str):
        """Load embedding timing report"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.embed_data = json.load(f)
            print(f"✅ Loaded embedding report: {file_path}")
        except Exception as e:
            print(f"❌ Failed to load embedding report: {e}")
    
    def analyze_llm_performance(self) -> Dict[str, Any]:
        """Phân tích chi tiết LLM performance"""
        if not self.llm_data:
            return {}
        
        call_history = self.llm_data.get("call_history", [])
        if not call_history:
            return {}
        
        # Extract timing data
        times = [call["call_time"] for call in call_history if call.get("call_time")]
        tokens = [call["total_tokens"] for call in call_history if call.get("total_tokens")]
        speeds = [call["tokens_per_second"] for call in call_history if call.get("tokens_per_second")]
        
        # Analyze by operation type
        operations = {}
        for call in call_history:
            op_type = call.get("operation_type", "unknown")
            if op_type not in operations:
                operations[op_type] = []
            operations[op_type].append(call)
        
        # Calculate statistics
        analysis = {
            "total_calls": len(call_history),
            "total_time": sum(times),
            "avg_time": statistics.mean(times) if times else 0,
            "median_time": statistics.median(times) if times else 0,
            "min_time": min(times) if times else 0,
            "max_time": max(times) if times else 0,
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
            "avg_tokens": statistics.mean(tokens) if tokens else 0,
            "avg_speed": statistics.mean(speeds) if speeds else 0,
            "operations": {}
        }
        
        # Operation-specific analysis
        for op_type, calls in operations.items():
            op_times = [call["call_time"] for call in calls if call.get("call_time")]
            op_tokens = [call["total_tokens"] for call in calls if call.get("total_tokens")]
            
            analysis["operations"][op_type] = {
                "count": len(calls),
                "avg_time": statistics.mean(op_times) if op_times else 0,
                "avg_tokens": statistics.mean(op_tokens) if op_tokens else 0,
                "total_time": sum(op_times) if op_times else 0
            }
        
        return analysis
    
    def analyze_embedding_performance(self) -> Dict[str, Any]:
        """Phân tích chi tiết embedding performance"""
        if not self.embed_data:
            return {}
        
        call_history = self.embed_data.get("call_history", [])
        if not call_history:
            return {}
        
        # Extract performance data
        times = [call["call_time"] for call in call_history if call.get("call_time")]
        texts = [call["texts_count"] for call in call_history if call.get("texts_count")]
        chars = [call["total_chars"] for call in call_history if call.get("total_chars")]
        tokens = [call["tokens_used"] for call in call_history if call.get("tokens_used")]
        text_speeds = [call["texts_per_second"] for call in call_history if call.get("texts_per_second")]
        
        # Batch size analysis
        batch_sizes = [call["batch_size"] for call in call_history if call.get("batch_size")]
        
        analysis = {
            "total_calls": len(call_history),
            "total_time": sum(times),
            "total_texts": sum(texts),
            "total_chars": sum(chars),
            "total_tokens": sum(tokens),
            "avg_time": statistics.mean(times) if times else 0,
            "avg_texts_per_call": statistics.mean(texts) if texts else 0,
            "avg_chars_per_call": statistics.mean(chars) if chars else 0,
            "avg_tokens_per_call": statistics.mean(tokens) if tokens else 0,
            "avg_text_speed": statistics.mean(text_speeds) if text_speeds else 0,
            "avg_batch_size": statistics.mean(batch_sizes) if batch_sizes else 0,
            "min_time": min(times) if times else 0,
            "max_time": max(times) if times else 0,
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0
        }
        
        return analysis
    
    def generate_performance_charts(self, output_dir: str = "."):
        """Generate performance charts"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('LightRAG Performance Analysis', fontsize=16, fontweight='bold')
            
            # LLM Analysis
            if self.llm_data and self.llm_data.get("call_history"):
                llm_calls = self.llm_data["call_history"]
                
                # 1. LLM Response Times
                times = [call["call_time"] for call in llm_calls if call.get("call_time")]
                axes[0, 0].hist(times, bins=20, alpha=0.7, color='blue', edgecolor='black')
                axes[0, 0].set_title('LLM Response Time Distribution')
                axes[0, 0].set_xlabel('Time (seconds)')
                axes[0, 0].set_ylabel('Frequency')
                
                # 2. LLM Tokens vs Time
                tokens = [call["total_tokens"] for call in llm_calls if call.get("total_tokens")]
                if len(times) == len(tokens):
                    axes[0, 1].scatter(tokens, times, alpha=0.6, color='blue')
                    axes[0, 1].set_title('LLM Tokens vs Response Time')
                    axes[0, 1].set_xlabel('Total Tokens')
                    axes[0, 1].set_ylabel('Time (seconds)')
                
                # 3. LLM Speed over time
                call_numbers = [call["call_number"] for call in llm_calls]
                speeds = [call["tokens_per_second"] for call in llm_calls if call.get("tokens_per_second")]
                if speeds:
                    axes[0, 2].plot(call_numbers[:len(speeds)], speeds, marker='o', color='blue')
                    axes[0, 2].set_title('LLM Speed Over Time')
                    axes[0, 2].set_xlabel('Call Number')
                    axes[0, 2].set_ylabel('Tokens/Second')
            
            # Embedding Analysis
            if self.embed_data and self.embed_data.get("call_history"):
                embed_calls = self.embed_data["call_history"]
                
                # 4. Embedding Response Times
                times = [call["call_time"] for call in embed_calls if call.get("call_time")]
                axes[1, 0].hist(times, bins=20, alpha=0.7, color='green', edgecolor='black')
                axes[1, 0].set_title('Embedding Response Time Distribution')
                axes[1, 0].set_xlabel('Time (seconds)')
                axes[1, 0].set_ylabel('Frequency')
                
                # 5. Batch Size vs Time
                batch_sizes = [call["batch_size"] for call in embed_calls if call.get("batch_size")]
                embed_times = [call["call_time"] for call in embed_calls if call.get("call_time")]
                if len(batch_sizes) == len(embed_times):
                    axes[1, 1].scatter(batch_sizes, embed_times, alpha=0.6, color='green')
                    axes[1, 1].set_title('Embedding Batch Size vs Time')
                    axes[1, 1].set_xlabel('Batch Size')
                    axes[1, 1].set_ylabel('Time (seconds)')
                
                # 6. Embedding Speed over time
                call_numbers = [call["call_number"] for call in embed_calls]
                text_speeds = [call["texts_per_second"] for call in embed_calls if call.get("texts_per_second")]
                if text_speeds:
                    axes[1, 2].plot(call_numbers[:len(text_speeds)], text_speeds, marker='o', color='green')
                    axes[1, 2].set_title('Embedding Speed Over Time')
                    axes[1, 2].set_xlabel('Call Number')
                    axes[1, 2].set_ylabel('Texts/Second')
            
            plt.tight_layout()
            chart_file = f"{output_dir}/performance_analysis.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Performance charts saved: {chart_file}")
            
        except ImportError:
            print("⚠️  matplotlib/seaborn not available. Install with: pip install matplotlib seaborn")
        except Exception as e:
            print(f"❌ Error generating charts: {e}")
    
    def generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # LLM Recommendations
        if self.llm_data:
            llm_analysis = self.analyze_llm_performance()
            
            if llm_analysis.get("avg_time", 0) > 5.0:
                recommendations.append(
                    "🔤 LLM: High average response time (>5s). Consider using a faster model or optimizing prompts."
                )
            
            if llm_analysis.get("std_dev", 0) > 2.0:
                recommendations.append(
                    "🔤 LLM: High variance in response times. Check for network issues or consider request batching."
                )
            
            # Operation-specific recommendations
            operations = llm_analysis.get("operations", {})
            for op_type, stats in operations.items():
                if stats["avg_time"] > 10.0:
                    recommendations.append(
                        f"🔤 LLM: {op_type} operations are slow ({stats['avg_time']:.1f}s avg). Consider optimization."
                    )
        
        # Embedding Recommendations
        if self.embed_data:
            embed_analysis = self.analyze_embedding_performance()
            
            if embed_analysis.get("avg_text_speed", 0) < 10:
                recommendations.append(
                    "🔢 Embedding: Low throughput (<10 texts/s). Consider increasing batch sizes."
                )
            
            if embed_analysis.get("avg_batch_size", 0) < 5:
                recommendations.append(
                    "🔢 Embedding: Small batch sizes detected. Batching texts can improve efficiency."
                )
            
            if embed_analysis.get("avg_time", 0) > 3.0:
                recommendations.append(
                    "🔢 Embedding: High average response time (>3s). Consider using a faster embedding model."
                )
        
        # General recommendations
        if not recommendations:
            recommendations.append("✅ Performance looks good! No specific optimizations needed.")
        
        return recommendations
    
    def export_analysis_report(self, output_file: str = None):
        """Export comprehensive analysis report"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"timing_analysis_report_{timestamp}.json"
        
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "llm_analysis": self.analyze_llm_performance(),
            "embedding_analysis": self.analyze_embedding_performance(),
            "recommendations": self.generate_recommendations(),
            "summary": self.generate_summary()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Analysis report exported: {output_file}")
        return output_file
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate performance summary"""
        summary = {
            "total_operations": 0,
            "total_time": 0,
            "overall_efficiency": "unknown"
        }
        
        # LLM Summary
        if self.llm_data:
            llm_analysis = self.analyze_llm_performance()
            summary["llm"] = {
                "calls": llm_analysis.get("total_calls", 0),
                "total_time": llm_analysis.get("total_time", 0),
                "avg_time": llm_analysis.get("avg_time", 0),
                "avg_speed": llm_analysis.get("avg_speed", 0)
            }
            summary["total_operations"] += llm_analysis.get("total_calls", 0)
            summary["total_time"] += llm_analysis.get("total_time", 0)
        
        # Embedding Summary
        if self.embed_data:
            embed_analysis = self.analyze_embedding_performance()
            summary["embedding"] = {
                "calls": embed_analysis.get("total_calls", 0),
                "total_time": embed_analysis.get("total_time", 0),
                "avg_time": embed_analysis.get("avg_time", 0),
                "avg_speed": embed_analysis.get("avg_text_speed", 0)
            }
            summary["total_operations"] += embed_analysis.get("total_calls", 0)
            summary["total_time"] += embed_analysis.get("total_time", 0)
        
        # Overall efficiency rating
        if summary["total_time"] > 0:
            ops_per_second = summary["total_operations"] / summary["total_time"]
            if ops_per_second > 2:
                summary["overall_efficiency"] = "excellent"
            elif ops_per_second > 1:
                summary["overall_efficiency"] = "good"
            elif ops_per_second > 0.5:
                summary["overall_efficiency"] = "average"
            else:
                summary["overall_efficiency"] = "needs_improvement"
        
        return summary
    
    def print_analysis(self):
        """Print comprehensive analysis to console"""
        print("\n" + "="*80)
        print("📊 LIGHTRAG TIMING ANALYSIS REPORT")
        print("="*80)
        
        # LLM Analysis
        if self.llm_data:
            llm_analysis = self.analyze_llm_performance()
            print(f"\n🔤 LLM PERFORMANCE ANALYSIS:")
            print(f"   📊 Total Calls: {llm_analysis['total_calls']}")
            print(f"   ⏱️  Total Time: {llm_analysis['total_time']:.2f}s")
            print(f"   📈 Average Time: {llm_analysis['avg_time']:.2f}s")
            print(f"   📊 Median Time: {llm_analysis['median_time']:.2f}s")
            print(f"   📏 Range: {llm_analysis['min_time']:.2f}s - {llm_analysis['max_time']:.2f}s")
            print(f"   📐 Std Dev: {llm_analysis['std_dev']:.2f}s")
            print(f"   ⚡ Average Speed: {llm_analysis['avg_speed']:.1f} tokens/s")
            print(f"   🔢 Average Tokens: {llm_analysis['avg_tokens']:.0f}")
            
            print(f"\n   📋 By Operation Type:")
            for op_type, stats in llm_analysis["operations"].items():
                print(f"      • {op_type}: {stats['count']} calls, "
                      f"{stats['avg_time']:.2f}s avg, "
                      f"{stats['avg_tokens']:.0f} tokens avg")
        
        # Embedding Analysis
        if self.embed_data:
            embed_analysis = self.analyze_embedding_performance()
            print(f"\n🔢 EMBEDDING PERFORMANCE ANALYSIS:")
            print(f"   📊 Total Calls: {embed_analysis['total_calls']}")
            print(f"   ⏱️  Total Time: {embed_analysis['total_time']:.2f}s")
            print(f"   📈 Average Time: {embed_analysis['avg_time']:.2f}s")
            print(f"   📏 Range: {embed_analysis['min_time']:.2f}s - {embed_analysis['max_time']:.2f}s")
            print(f"   📐 Std Dev: {embed_analysis['std_dev']:.2f}s")
            print(f"   ⚡ Average Speed: {embed_analysis['avg_text_speed']:.1f} texts/s")
            print(f"   📄 Average Batch Size: {embed_analysis['avg_batch_size']:.1f}")
            print(f"   🔤 Total Texts: {embed_analysis['total_texts']:,}")
            print(f"   📝 Total Characters: {embed_analysis['total_chars']:,}")
            print(f"   🎯 Total Tokens: {embed_analysis['total_tokens']:,}")
        
        # Recommendations
        recommendations = self.generate_recommendations()
        print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        # Summary
        summary = self.generate_summary()
        print(f"\n📈 OVERALL SUMMARY:")
        print(f"   🎯 Total Operations: {summary['total_operations']}")
        print(f"   ⏱️  Total Time: {summary['total_time']:.2f}s")
        print(f"   📊 Efficiency Rating: {summary['overall_efficiency'].replace('_', ' ').title()}")
        
        print("="*80)


def main():
    """Main function với command line interface"""
    parser = argparse.ArgumentParser(description="Analyze LightRAG timing data")
    parser.add_argument("--llm-report", help="Path to LLM timing report JSON file")
    parser.add_argument("--embed-report", help="Path to embedding timing report JSON file")
    parser.add_argument("--output-dir", default=".", help="Output directory for charts and reports")
    parser.add_argument("--charts", action="store_true", help="Generate performance charts")
    parser.add_argument("--export", help="Export analysis report to file")
    
    args = parser.parse_args()
    
    if not args.llm_report and not args.embed_report:
        print("❌ Please provide at least one report file (--llm-report or --embed-report)")
        return
    
    # Initialize analyzer
    analyzer = TimingAnalyzer(args.llm_report, args.embed_report)
    
    # Print analysis
    analyzer.print_analysis()
    
    # Generate charts if requested
    if args.charts:
        analyzer.generate_performance_charts(args.output_dir)
    
    # Export report if requested
    if args.export:
        analyzer.export_analysis_report(args.export)
    else:
        # Auto-export with timestamp
        analyzer.export_analysis_report()


if __name__ == "__main__":
    main()