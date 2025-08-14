#!/usr/bin/env python3
"""
Example of how the workflow generator would use the tool information functions.
This demonstrates the separation between tool planning and tool execution.
"""

from utils import (
    get_tools_for_generation,
    get_tools_by_category,
    get_storage_dependent_tools,
    get_api_key_dependent_tools
)

def plan_workflow_with_tools():
    """
    Example of how the workflow generator would plan workflows using tool information.
    This happens during the generation phase, before actual tool execution.
    """
    print("🔧 Workflow Generation Phase - Tool Planning")
    print("=" * 60)
    
    # Get all available tools for planning
    all_tools = get_tools_for_generation()
    print(f"📋 Total available tools: {len(all_tools)}")
    
    # Plan different types of workflows based on available capabilities
    print("\n🎯 Planning Search-Based Workflow:")
    search_tools = get_tools_by_category("search")
    for tool in search_tools:
        print(f"   ✓ {tool['name']}: {tool['description']}")
        print(f"     Capabilities: {', '.join(tool['capabilities'])}")
    
    print("\n🗄️ Planning Database Workflow:")
    database_tools = get_tools_by_category("database")
    for tool in database_tools:
        print(f"   ✓ {tool['name']}: {tool['description']}")
        print(f"     Capabilities: {', '.join(tool['capabilities'])}")
    
    print("\n🎨 Planning Image Generation Workflow:")
    image_tools = get_tools_by_category("image_generation")
    for tool in image_tools:
        print(f"   ✓ {tool['name']}: {tool['description']}")
        if tool.get('requires_api_key'):
            print(f"     Requires API Key: {tool['requires_api_key']}")
    
    print("\n📁 Planning File Processing Workflow:")
    storage_tools = get_storage_dependent_tools()
    for tool in storage_tools:
        print(f"   ✓ {tool['name']}: {tool['description']}")
        print(f"     Storage Required: {tool.get('requires_storage', False)}")
    
    print("\n🔑 API Key Requirements:")
    api_tools = get_api_key_dependent_tools()
    for tool in api_tools:
        print(f"   ✓ {tool['name']}: {tool['requires_api_key']}")

def analyze_tool_requirements():
    """
    Example of analyzing tool requirements for workflow planning.
    """
    print("\n📊 Tool Requirements Analysis")
    print("=" * 60)
    
    all_tools = get_tools_for_generation()
    
    # Count tools by category
    categories = {}
    storage_required = 0
    api_keys_required = set()
    
    for tool in all_tools:
        category = tool.get('category', 'unknown')
        categories[category] = categories.get(category, 0) + 1
        
        if tool.get('requires_storage', False):
            storage_required += 1
            
        if tool.get('requires_api_key'):
            api_keys_required.add(tool['requires_api_key'])
    
    print("📈 Tool Distribution by Category:")
    for category, count in categories.items():
        print(f"   {category}: {count} tools")
    
    print(f"\n💾 Storage-Dependent Tools: {storage_required}")
    print(f"🔑 API Keys Required: {', '.join(api_keys_required) if api_keys_required else 'None'}")

def generate_workflow_suggestions():
    """
    Example of generating workflow suggestions based on available tools.
    """
    print("\n💡 Workflow Suggestions Based on Available Tools")
    print("=" * 60)
    
    # Get tools by capability
    search_tools = get_tools_by_category("search")
    database_tools = get_tools_by_category("database")
    image_tools = get_tools_by_category("image_generation")
    
    if search_tools and database_tools:
        print("🔍 Suggested Workflow: Research & Data Collection")
        print("   Use search tools to gather information, then store in database")
    
    if image_tools:
        print("🎨 Suggested Workflow: Content Creation")
        print("   Generate images based on user requirements")
    
    if database_tools:
        print("📊 Suggested Workflow: Data Analysis")
        print("   Use database tools for data processing and analysis")

if __name__ == "__main__":
    print("🚀 Workflow Generator Tool Planning Example")
    print("This demonstrates how tools are planned during generation phase")
    
    plan_workflow_with_tools()
    analyze_tool_requirements()
    generate_workflow_suggestions()
    
    print("\n" + "=" * 60)
    print("✅ Tool planning completed!")
    print("\n💡 Key Benefits:")
    print("   - No tool instances created during planning")
    print("   - Fast workflow generation without dependencies")
    print("   - Clear understanding of tool capabilities")
    print("   - Easy to filter tools by requirements")
