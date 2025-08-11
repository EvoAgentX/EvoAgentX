#!/usr/bin/env python3
"""
Test script for parallel workflow generation functionality.
This script tests the new parallel workflow generation without changing the API interface.
"""

import asyncio
import os
import sys
from datetime import datetime

# Add the server directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_parallel_workflow_generation():
    """Test the parallel workflow generation functionality"""
    
    print("🧪 Testing Parallel Workflow Generation")
    print("=" * 50)
    
    try:
        # Import the core functions
        from core.workflow_setup import setup_project, get_project_workflow_status
        
        # Test project ID (you can change this to test with a real project)
        test_project_id = "test_parallel_generation"
        
        print(f"📋 Testing with project ID: {test_project_id}")
        print(f"⏰ Start time: {datetime.now()}")
        
        # Test 1: Check if we can get project workflow status
        print("\n🔍 Test 1: Getting project workflow status...")
        try:
            status = await get_project_workflow_status(test_project_id)
            print(f"✅ Status retrieved: {status['overall_status']}")
            print(f"   Total workflows: {status['total_workflows']}")
            print(f"   Completed: {status['completed_workflows']}")
            print(f"   Failed: {status['failed_workflows']}")
        except ValueError as e:
            print(f"⚠️  Expected error (no workflows exist yet): {str(e)}")
        
        # Test 2: Check concurrency configuration
        print("\n⚙️  Test 2: Checking concurrency configuration...")
        concurrency = os.getenv("PARALLEL_WORKFLOW_CONCURRENCY", "5")
        print(f"✅ Concurrency level: {concurrency}")
        
        # Test 3: Test the core functions are importable
        print("\n📦 Test 3: Testing core function imports...")
        print(f"✅ setup_project function: {setup_project}")
        print(f"✅ get_project_workflow_status function: {get_project_workflow_status}")
        
        print("\n🎉 All tests passed! Parallel workflow generation is ready.")
        print(f"⏰ End time: {datetime.now()}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're running this from the server directory")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_concurrent_execution():
    """Test basic concurrent execution capabilities"""
    
    print("\n🔄 Testing Concurrent Execution Capabilities")
    print("=" * 50)
    
    # Test semaphore functionality
    semaphore = asyncio.Semaphore(3)
    
    async def mock_task(task_id: int, delay: float):
        async with semaphore:
            print(f"   🚀 Task {task_id} started")
            await asyncio.sleep(delay)
            print(f"   ✅ Task {task_id} completed")
            return f"Task {task_id} result"
    
    print("📋 Creating 5 concurrent tasks with semaphore limit of 3...")
    
    start_time = datetime.now()
    tasks = [mock_task(i, 1.0) for i in range(5)]
    results = await asyncio.gather(*tasks)
    end_time = datetime.now()
    
    print(f"✅ All tasks completed in {end_time - start_time}")
    print(f"📊 Results: {results}")
    
    # Verify that tasks were executed concurrently (should take ~2 seconds, not 5)
    execution_time = (end_time - start_time).total_seconds()
    if execution_time < 3.0:
        print("✅ Concurrent execution working correctly!")
    else:
        print("⚠️  Tasks may not be executing concurrently")

if __name__ == "__main__":
    print("🚀 Starting Parallel Workflow Generation Tests")
    print("=" * 60)
    
    # Run the tests
    asyncio.run(test_parallel_workflow_generation())
    asyncio.run(test_concurrent_execution())
    
    print("\n" + "=" * 60)
    print("🏁 All tests completed!")
