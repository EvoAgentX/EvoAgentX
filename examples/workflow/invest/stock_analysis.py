<<<<<<< HEAD
=======
# Main function to run

>>>>>>> afda1368dac6a538209e7ab5b8f1fe679c3c0272
import sys
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import time

# Set matplotlib backend to avoid threading issues
import matplotlib
matplotlib.use('Agg')

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
<<<<<<< HEAD
from .catl_data_functions import fetch_stock_data
from .stock_chart_tools import generate_stock_charts
=======
from catl_data_functions import fetch_stock_data
from stock_chart_tools import generate_stock_charts
>>>>>>> afda1368dac6a538209e7ab5b8f1fe679c3c0272
# EvoAgentX imports
from evoagentx.models import OpenAILLMConfig, OpenAILLM, OpenRouterConfig, OpenRouterLLM
from evoagentx.workflow import WorkFlowGraph, WorkFlow, WorkFlowGenerator
from evoagentx.agents import AgentManager
from evoagentx.tools import StorageToolkit, CMDToolkit

load_dotenv()
<<<<<<< HEAD
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPEN_ROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
=======

# Read API keys from files
def read_api_key(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

OPENAI_API_KEY = read_api_key("openai_api_key.txt") or os.getenv("OPENAI_API_KEY")
OPEN_ROUTER_API_KEY = read_api_key("openrouter_api_key.txt") or os.getenv("OPENROUTER_API_KEY")
>>>>>>> afda1368dac6a538209e7ab5b8f1fe679c3c0272

# Fixed variables and paths
available_funds = 100000
current_positions = 500
average_price = 280
position_type = "call"
report_date = datetime.now().strftime('%Y-%m-%d')
llm = OpenAILLM(config=OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=True, max_tokens=16000))
tools = [StorageToolkit(), CMDToolkit()]

# Path to the workflow module (should be pre-generated)
<<<<<<< HEAD
module_save_path = "examples/workflow/invest/invest_demo_4o_mini.json"
=======
module_save_path = "invest_demo_4o_mini_v1.json"
>>>>>>> afda1368dac6a538209e7ab5b8f1fe679c3c0272

# Workflow generation goal (commented out for future use)
WORKFLOW_GOAL = """Create a daily trading decision workflow for A-share stocks.

## Workflow Overview:
A multi-step workflow for daily trading decisions with fixed capital, making trading decisions based on market data and current positions.

## Task Description:
**Name:** daily_trading_decision
**Description:** A comprehensive trading decision system that analyzes market data and generates daily trading operations with detailed analysis.

## Input:
- **goal** (string): Contains stock code, available funds, current positions, data folder path, output file path, and optional past report path

## Output:
- **trading_report** (string): A comprehensive daily trading report with complete analysis

## Analysis Requirements:
The workflow should analyze three key aspects of the stock:

1. **Background Analysis**: Market environment, industry trends, news sentiment, expert opinions, economic factors, and regulatory environment that affect stock prices
2. **Price Analysis**: Historical price patterns, technical indicators, support/resistance levels, and trading volume analysis
3. **Performance Review**: Past trading decisions, performance evaluation, and lessons learned from previous reports

## Workflow Structure:
- Start with file discovery to identify and categorize available data sources
- Perform the three analyses in parallel where possible for efficiency
- Compile all findings into a comprehensive trading report

## Agent Guidelines:
- Agents should use appropriate tools to discover and read files from the data folder
- Each analysis should focus on its specific domain without overlap
- Agents should filter out irrelevant files and focus on data relevant to their analysis
- All analysis must be based on actual data from files - no fake or estimated data
- Present complete data without omissions or truncations

## Report Structure:
The final report should include:
1. **Background Analysis**: Market environment and external factors
2. **Price Analysis**: Technical patterns and indicators
3. **Performance Review**: Historical performance and lessons learned
4. **Trading Recommendations**: Specific buy/sell/hold decisions with quantities and prices

## Critical Requirements:
- Base all analysis on actual data read from files
- If no relevant files are found, report this clearly and do not make up data
- Provide specific trading recommendations with quantities and price targets
- Consider current positions and available capital in decision making
- Structure the report with clear sections and data tables
- Return complete analysis without summarization
"""


def get_directories(stock_code, timestamp):
    """Get directory paths for a given stock code and timestamp"""
<<<<<<< HEAD
    # Use the output directory under examples/workflow/invest/
    base_dir = Path(f"./examples/workflow/invest/output/{stock_code}")
    data_dir = base_dir / timestamp / "data"
    report_dir = base_dir / "reports"
=======
    base_dir = Path(f"./{stock_code}")
    data_dir = base_dir / timestamp / "data"
    report_dir = base_dir  / "reports"
>>>>>>> afda1368dac6a538209e7ab5b8f1fe679c3c0272
    graphs_dir = base_dir / timestamp / "graphs"
    return base_dir, data_dir, report_dir, graphs_dir


def check_data_exists(data_dir):
    """Check if data files already exist in the data directory"""
    if not data_dir.exists():
        return False
    
    # Check for common data file patterns
    expected_files = [
        "stock_daily_catl_*.csv",
        "china_cpi_*.csv", 
        "china_gdp_yearly_*.csv",
        "industry_fund_flow_*.csv",
        "stock_news_catl_*.csv",
        "market_summary_sse_*.csv",
        "market_indices_*.csv",
        "option_volatility_50etf_*.csv",
        "institution_recommendation_catl_*.csv"
    ]
    
    existing_files = list(data_dir.glob("*.csv"))
    if len(existing_files) >= 5:  # At least 5 data files exist
        print(f"✅ 数据文件已存在: {data_dir}")
        print(f"   发现 {len(existing_files)} 个数据文件")
        return True
    
    return False


def check_charts_exist(graphs_dir, stock_code):
    """Check if chart files already exist"""
    if not graphs_dir.exists():
        return False
    
    expected_charts = [
        f"{stock_code}_technical_charts.png",
        f"{stock_code}_candlestick_chart.png"
    ]
    
    existing_charts = [f.name for f in graphs_dir.glob("*.png")]
    if all(chart in existing_charts for chart in expected_charts):
        print(f"✅ 图表文件已存在: {graphs_dir}")
        print(f"   发现 {len(existing_charts)} 个图表文件")
        return True
    
    return False


def generate_workflow():
    """Generate a new workflow (commented out for future use)"""
    # Uncomment the following lines to generate a new workflow
    wf_generator = WorkFlowGenerator(llm=llm, tools=tools)
    workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=WORKFLOW_GOAL, retry=5)
    workflow_graph.save_module(module_save_path)
    return workflow_graph


def execute_workflow(stock_code, data_dir, report_dir, timestamp):
    """Execute the workflow with the given parameters"""
    try:
        # Load workflow graph
        workflow_graph: WorkFlowGraph = WorkFlowGraph.from_file(module_save_path)
        agent_manager = AgentManager(tools=tools)
<<<<<<< HEAD
        
        # Override any placeholder LLM configs in the workflow with our actual config
        for node in workflow_graph.nodes:
            if node.agents:
                for agent in node.agents:
                    if isinstance(agent, dict) and "llm_config" in agent:
                        # Replace any placeholder API keys with our actual config
                        agent["llm_config"] = llm.config.to_dict()
        
=======
>>>>>>> afda1368dac6a538209e7ab5b8f1fe679c3c0272
        agent_manager.add_agents_from_workflow(workflow_graph, llm_config=llm.config)
        workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
        workflow.init_module()

        # Construct the goal string
<<<<<<< HEAD
        comprehensive_report_file = report_dir / f"comprehensive_report_{stock_code}_{timestamp}.md"
        past_report = report_dir / f"comprehensive_report_{stock_code}_{timestamp}_previous.md"
=======
        output_file = report_dir / f"text_report_{stock_code}_{timestamp}.md"
        past_report = report_dir / f"text_report_{stock_code}_{timestamp}_previous.md"
>>>>>>> afda1368dac6a538209e7ab5b8f1fe679c3c0272
        
        goal = f"""I need a daily trading decision for stock {stock_code}.
Available funds: {available_funds} RMB
Current positions: {current_positions} shares of {stock_code} at average price {average_price} RMB
Date: {report_date}
Type of position: {position_type}
Data folder: {data_dir}
Past report folder: {past_report}

Please read ALL files in the data folder and generate a comprehensive trading decision report in Chinese based on real data. Return the complete content.
"""

<<<<<<< HEAD
        # Execute the workflow
        workflow.execute({"goal": goal})
        
        # Get the raw output from the workflow environment instead of using output extraction
        try:
            # Get the final task's messages from the environment
            end_tasks = workflow.graph.find_end_nodes()
            if end_tasks:
                final_task = end_tasks[0]  # Get the first end task
                final_messages = workflow.environment.get_task_messages(tasks=final_task, n=1)
                
                if final_messages:
                    # Get the raw content from the final message
                    raw_output = str(final_messages[0].content)
                    
                    # Check if the output is JSON and extract the markdown content
                    if raw_output.strip().startswith('{'):
                        try:
                            import json
                            json_data = json.loads(raw_output)
                            if 'comprehensive_report' in json_data:
                                # Extract the markdown content from JSON
                                markdown_content = json_data['comprehensive_report']
                                # Save the clean markdown content
                                with open(comprehensive_report_file, "w", encoding="utf-8") as f:
                                    f.write(markdown_content)
                                print(f"✅ Comprehensive report saved to: {comprehensive_report_file}")
                            else:
                                # Fallback: save the raw content
                                with open(comprehensive_report_file, "w", encoding="utf-8") as f:
                                    f.write(raw_output)
                                print(f"✅ Comprehensive report saved to: {comprehensive_report_file}")
                        except json.JSONDecodeError:
                            # If JSON parsing fails, save the raw content
                            with open(comprehensive_report_file, "w", encoding="utf-8") as f:
                                f.write(raw_output)
                            print(f"✅ Comprehensive report saved to: {comprehensive_report_file}")
                    else:
                        # If it's not JSON, save the raw content directly
                        with open(comprehensive_report_file, "w", encoding="utf-8") as f:
                            f.write(raw_output)
                        print(f"✅ Comprehensive report saved to: {comprehensive_report_file}")
                    
                else:
                    # Fallback: get all messages and use the last one
                    all_messages = workflow.environment.get()
                    if all_messages:
                        raw_output = str(all_messages[-1].content)
                        
                        # Check if the output is JSON and extract the markdown content
                        if raw_output.strip().startswith('{'):
                            try:
                                import json
                                json_data = json.loads(raw_output)
                                if 'comprehensive_report' in json_data:
                                    # Extract the markdown content from JSON
                                    markdown_content = json_data['comprehensive_report']
                                    # Save the clean markdown content
                                    with open(comprehensive_report_file, "w", encoding="utf-8") as f:
                                        f.write(markdown_content)
                                    print(f"✅ Comprehensive report saved to: {comprehensive_report_file}")
                                else:
                                    # Fallback: save the raw content
                                    with open(comprehensive_report_file, "w", encoding="utf-8") as f:
                                        f.write(raw_output)
                                    print(f"✅ Comprehensive report saved to: {comprehensive_report_file}")
                            except json.JSONDecodeError:
                                # If JSON parsing fails, save the raw content
                                with open(comprehensive_report_file, "w", encoding="utf-8") as f:
                                    f.write(raw_output)
                                print(f"✅ Comprehensive report saved to: {comprehensive_report_file}")
                        else:
                            # If it's not JSON, save the raw content directly
                            with open(comprehensive_report_file, "w", encoding="utf-8") as f:
                                f.write(raw_output)
                            print(f"✅ Comprehensive report saved to: {comprehensive_report_file}")
                    else:
                        print("❌ No messages found in workflow environment")
                        
            else:
                print("❌ No end tasks found in workflow")
                
        except Exception as e:
            print(f"Error saving comprehensive report: {e}")
            import traceback
            traceback.print_exc()
            
=======
        output = workflow.execute({"goal": goal})
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Trading decision report saved to: {output_file}")
            # # Also save a backup
            # with open(report_dir / f"text_report_{stock_code}_{timestamp}_back.md", "w", encoding="utf-8") as f:
            #     f.write(output)
        except Exception as e:
            print(f"Error saving report: {e}")
>>>>>>> afda1368dac6a538209e7ab5b8f1fe679c3c0272
    except Exception as e:
        print(f"Error executing workflow: {e}")
        import traceback
        traceback.print_exc()


def generate_html_report(stock_code, base_dir, report_dir, graphs_dir, timestamp):
<<<<<<< HEAD
    """Generate HTML report from comprehensive markdown and charts"""
=======
    """Generate HTML report from markdown and charts"""
>>>>>>> afda1368dac6a538209e7ab5b8f1fe679c3c0272
    try:
        # Import the HTML generator
        from html_report_generator import HTMLGenerator
        
<<<<<<< HEAD
        # Define file paths - use comprehensive report as primary
        comprehensive_report_file = report_dir / f"comprehensive_report_{stock_code}_{timestamp}.md"
=======
        # Define file paths
        md_file = report_dir / f"text_report_{stock_code}_{timestamp}.md"
>>>>>>> afda1368dac6a538209e7ab5b8f1fe679c3c0272
        html_output = base_dir/ datetime.now().strftime('%Y%m%d') / "html_report" / f"report_{stock_code}_{timestamp}.html"
        
        # Find chart files
        technical_chart = graphs_dir / f"{stock_code}_technical_charts.png"
        price_volume_chart = graphs_dir / f"{stock_code}_candlestick_chart.png"
        
<<<<<<< HEAD
        # Check if comprehensive report exists (primary file)
        if not comprehensive_report_file.exists():
            print(f"❌ Comprehensive report file not found: {comprehensive_report_file}")
=======
        # Check if markdown file exists
        if not md_file.exists():
            print(f"❌ Markdown file not found: {md_file}")
>>>>>>> afda1368dac6a538209e7ab5b8f1fe679c3c0272
            return False
        
        # Check if charts exist
        if not technical_chart.exists():
            print(f"⚠️  Technical chart not found: {technical_chart}")
            technical_chart = ""
        
        if not price_volume_chart.exists():
            print(f"⚠️  Price/volume chart not found: {price_volume_chart}")
            price_volume_chart = ""
        
<<<<<<< HEAD
        # Generate HTML report with comprehensive report
        print(f"[4] 生成HTML报告: {html_output}")
        generator = HTMLGenerator(str(html_output))
        output_file = generator.generate_report(
            str(comprehensive_report_file), 
=======
        # Generate HTML report
        print(f"[4] 生成HTML报告: {html_output}")
        generator = HTMLGenerator(str(html_output))
        output_file = generator.generate_report(
            str(md_file), 
>>>>>>> afda1368dac6a538209e7ab5b8f1fe679c3c0272
            str(technical_chart) if technical_chart else "", 
            str(price_volume_chart) if price_volume_chart else ""
        )
        
        print(f"✅ HTML报告生成成功: {output_file}")
        print(f"📁 资源文件夹: {Path(output_file).parent / 'assets'}")
        print(f"🌐 在浏览器中打开HTML文件查看报告")
        
        return True
        
    except Exception as e:
        print(f"❌ HTML报告生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


<<<<<<< HEAD
def check_reports_exist(stock_code, report_dir, timestamp):
    """Check if comprehensive report exists"""
    comprehensive_report_file = report_dir / f"comprehensive_report_{stock_code}_{timestamp}.md"
    
    comprehensive_exists = comprehensive_report_file.exists()
    
    print(f"📊 报告生成状态:")
    if comprehensive_exists:
        print(f"   ✅ 综合分析报告: {comprehensive_report_file}")
    else:
        print(f"   ❌ 综合分析报告: {comprehensive_report_file}")
    
    return comprehensive_exists


=======
>>>>>>> afda1368dac6a538209e7ab5b8f1fe679c3c0272
def generate_html_from_existing_files(stock_code, timestamp=None):
    """Generate HTML report from existing markdown and chart files"""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d')
    
    base_dir, data_dir, report_dir, graphs_dir = get_directories(stock_code, timestamp)
    
    print(f"🔍 查找现有文件:")
    print(f"   报告目录: {report_dir}")
    print(f"   图表目录: {graphs_dir}")
    
    # Check if directories exist
    if not report_dir.exists():
        print(f"❌ 报告目录不存在: {report_dir}")
        return False
    
    if not graphs_dir.exists():
        print(f"⚠️  图表目录不存在: {graphs_dir}")
        graphs_dir = None
    
<<<<<<< HEAD
    # Check for comprehensive report
    comprehensive_report_file = report_dir / f"comprehensive_report_{stock_code}_{timestamp}.md"
    
    if comprehensive_report_file.exists():
        print(f"✅ 找到综合分析报告: {comprehensive_report_file}")
    else:
        print(f"❌ 未找到综合分析报告: {comprehensive_report_file}")
    
=======
>>>>>>> afda1368dac6a538209e7ab5b8f1fe679c3c0272
    return generate_html_report(stock_code, base_dir, report_dir, graphs_dir, timestamp)


def main():
    if len(sys.argv) < 2:
        stock_code = input("请输入股票代码 (如300750): ").strip()
    else:
        stock_code = sys.argv[1].strip()
    if not stock_code.isdigit():
        print("❌ 股票代码应为数字！")
        return
    # stock_code = "300750"
    
    timestamp = datetime.now().strftime('%Y%m%d')
    base_dir, data_dir, report_dir, graphs_dir = get_directories(stock_code, timestamp)
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    # Check and fetch data if needed
    if not check_data_exists(data_dir):
        print(f"\n[1] 拉取数据到: {data_dir}")
        fetch_stock_data(stock_code, output_dir=str(data_dir))
    else:
        print(f"\n[1] 跳过数据拉取 (数据已存在)")
    
    # Check and generate charts if needed
    if not check_charts_exist(graphs_dir, stock_code):
        print(f"[2] 生成图表到: {graphs_dir}")
        generate_stock_charts(stock_code, output_dir=str(graphs_dir))
    else:
        print(f"[2] 跳过图表生成 (图表已存在)")
    
    # === Workflow logic from workflow_invest.py ===
    print(f"[3] 生成报告到: {report_dir}")
<<<<<<< HEAD
    print(f"   将生成一个文件:")
    print(f"   - comprehensive_report_{stock_code}_{timestamp}.md (综合分析报告)")
    # generate_workflow(llm, tools)
    execute_workflow(stock_code, data_dir, report_dir, timestamp)
    
    # Check if comprehensive report was generated successfully
    comprehensive_exists = check_reports_exist(stock_code, report_dir, timestamp)
    
=======
    # generate_workflow(llm, tools)
    execute_workflow(stock_code, data_dir, report_dir, timestamp)
    
>>>>>>> afda1368dac6a538209e7ab5b8f1fe679c3c0272
    # === Generate HTML report ===
    print(f"\n[4] 生成HTML报告")
    html_success = generate_html_report(stock_code, base_dir, report_dir, graphs_dir, timestamp)
    
    if html_success:
        print("\n✅ 全部流程完成！包括HTML报告生成")
<<<<<<< HEAD
        print(f"📁 报告文件位置:")
        print(f"   - 综合分析报告: {report_dir}/comprehensive_report_{stock_code}_{timestamp}.md")
        print(f"   - HTML报告: {base_dir}/{timestamp}/html_report/report_{stock_code}_{timestamp}.html")
    else:
        print("\n✅ 主要流程完成！(HTML报告生成失败)")
        print(f"📁 报告文件位置:")
        print(f"   - 综合分析报告: {report_dir}/comprehensive_report_{stock_code}_{timestamp}.md")
=======
    else:
        print("\n✅ 主要流程完成！(HTML报告生成失败)")
>>>>>>> afda1368dac6a538209e7ab5b8f1fe679c3c0272

if __name__ == "__main__":
    main()
