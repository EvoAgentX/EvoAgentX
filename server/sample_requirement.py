SAMPLE_REQUIREMENT = """
 # 宠物管理网站 - 需求分析

 ## AI工作流决策
 **疾病原因分析**: 
 - 名称: disease_analysis
 - id: 550e8400-e29b-41d4-a716-446655440001
 - description: Analyze the pet's symptoms and provide possible causes of the disease
 - url: POST /api/ai/workflows/550e8400-e29b-41d4-a716-446655440001/execute
 - inputs: [
     { "name": "symptoms", "type": "string", "required": true, "description": "Information about the pet's symptoms" }
     { "name": "pet_info", "type": "object", "required": true, "description": "Information about the pet" }
   ]
 - outputs: [
     { "name": "possible_causes", "type": "array", "description": "Possible causes of the disease" }
   ]

 **治疗推荐**: 
 - name: treatment_recommendation
 - id: 550e8400-e29b-41d4-a716-446655440002
 - description: Recommend treatment options based on the diagnosis of the disease
 - POST /api/ai/workflows/550e8400-e29b-41d4-a716-446655440002/execute
 - inputs: [
     { "name": "diagnosis", "type": "string", "required": true, "description": "Diagnosis of the disease" }
   ]
 - outputs: [
     { "name": "treatment_options", "type": "array", "description": "Recommended treatment options" },
     { "name": "treatment_plan", "type": "string", "description": "Treatment plan" }
   ]

 ## 1. 系统架构图
 ```mermaid
 graph TB
     subgraph "前端层"
         React[React应用<br/>Vite + React]:::frontend
         PetManagement[宠物管理组件]:::frontend
         HealthRecords[健康记录管理组件]:::frontend
     end
     
     subgraph "后端层"
         Gateway[API网关<br/>NestJS]:::backend
         DynamicAPI[动态实体API]:::backend
         AIWorkflow[AI工作流API]:::backend
     end
     
     subgraph "数据层"
         Prisma[Prisma ORM]:::database
         DB[(PostgreSQL)]:::database
     end
     
     React --> Gateway
     PetManagement --> DynamicAPI
     HealthRecords --> DynamicAPI
     HealthRecords -.-> AIWorkflow
     
     Gateway --> DynamicAPI
     Gateway --> AIWorkflow
     
     DynamicAPI --> Prisma
     AIWorkflow --> Prisma
     Prisma --> DB

     classDef frontend fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000
     classDef backend fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000
     classDef database fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px,color:#000
 ```
 
 ## 2. 核心需求
 
 ### 2.1 宠物管理（动态实体）
 **实体定义**：`pets`
 ```json
 {
   "name": "pets",
   "type": "object",
   "properties": {
     "name": { "type": "string", "required": true },
     "species": { "type": "string", "required": true },
     "age": { "type": "number", "required": true },
     "owner_id": { "type": "string", "required": true }
   }
 }
 ```

 ### 2.2 健康记录（动态实体）
 **实体定义**：`health_records`
 ```json
 {
   "name": "health_records",
   "type": "object",
   "properties": {
     "pet_id": { "type": "string", "required": true },
     "date": { "type": "date", "required": true },
     "symptoms": { "type": "string", "required": true },
     "diagnosis": { "type": "string", "required": false },
     "treatment": { "type": "string", "required": false }
   }
 }
 ```

 ## 3. 前端组件
 - PetList: 显示和管理宠物
 - HealthRecordForm: 创建/编辑健康记录
 - DiseaseAnalysisButton: 触发疾病原因分析
 - TreatmentRecommendationButton: 触发治疗推荐

 ## 4. API操作
 **标准CRUD**： /api/apps/{appId}/entities/{pets|health_records}
 **AI功能**： /api/ai/workflows/{workflow-id}/execute
"""