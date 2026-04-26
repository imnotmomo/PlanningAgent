# Project Plan: TripWise Multi-Agent Travel Planning Agent

## Project Title

**TripWise: A Multi-Agent Travel Planning Assistant with Fine-Tuned Itinerary Generation**

## 1. Problem Statement

Travel planning usually requires users to combine information from many sources: destinations, attractions, restaurants, hotels, maps, transportation, weather, budgets, and personal preferences. A normal chatbot can generate a travel plan, but it may ignore constraints, create unrealistic schedules, hallucinate unavailable places, or fail to revise the plan correctly when the user changes requirements.

This project builds a **multi-agent travel planning system** that decomposes travel planning into specialized agents. Each agent handles one part of the planning process, and the system produces a realistic, constraint-aware, personalized travel itinerary.

## 2. Main Goal

The goal is to create a travel planning assistant that can:

1. Understand user preferences and constraints
2. Ask for missing information when necessary
3. Retrieve or use destination information
4. Group attractions by location and feasibility
5. Estimate budget and transportation needs
6. Generate a clean day-by-day itinerary
7. Check whether the plan is realistic
8. Revise the plan based on user feedback

Example user request:

> Plan a 5-day trip to Tokyo for two people. We like anime, food, shopping, and city views. Budget is medium and we prefer public transportation.

Expected output:

> A day-by-day itinerary with themes, morning/afternoon/evening activities, transportation notes, estimated cost, feasibility notes, backup options, and travel tips.

## 3. System Overview

The system uses a shared LLM backbone with different prompts for different agents. The agents do not need to be separate models. They can use the same LLM but behave differently because each one has a different role, prompt, tool access, and output format.

```text
User Query
   ↓
Preference Agent
   ↓
Missing Information Checker
   ↓
Research Agent
   ↓
Route Planning Agent
   ↓
Budget Agent
   ↓
Itinerary Generation Agent
   ↓
Critic / Feasibility Agent
   ↓
Final Itinerary
```

## 4. Multi-Agent Design

| Agent | Responsibility | Output |
|---|---|---|
| Preference Agent | Extract destination, dates, budget, travelers, interests, pace, and constraints | Structured JSON preferences |
| Missing Information Agent | Checks whether required information is missing | Missing fields or approval to continue |
| Research Agent | Finds or retrieves attractions, neighborhoods, events, and local context | Candidate places and notes |
| Route Planning Agent | Groups nearby attractions by day and estimates feasibility | Route groups |
| Budget Agent | Estimates food, transport, tickets, and activity costs | Budget summary |
| Itinerary Generation Agent | Produces the final day-by-day itinerary | Structured itinerary JSON |
| Critic Agent | Checks if the plan is too rushed, unrealistic, or violates constraints | Critique and suggested fixes |
| Revision Agent | Updates the itinerary when the user requests changes | Revised itinerary |

## 5. Recommended Architecture

```text
Frontend: Next.js + Tailwind CSS
   ↓
Backend: FastAPI
   ↓
Agent Orchestrator: LangGraph or custom state machine
   ↓
Agents:
   - Preference Agent
   - Research Agent
   - Route Agent
   - Budget Agent
   - Itinerary Agent
   - Critic Agent
   - Revision Agent
   ↓
Tools:
   - Search API
   - Maps / routing API
   - Weather API
   - Vector database
   - User profile memory
   ↓
LLM:
   - GPT-5 API or open-source model for orchestration
   - Fine-tuned 7B model for itinerary generation
```

## 6. Suggested Tech Stack

### Frontend

| Purpose | Tool |
|---|---|
| Web app | Next.js |
| Styling | Tailwind CSS |
| UI components | shadcn/ui |
| Icons | Lucide React |
| Budget charts | Recharts |

### Backend

| Purpose | Tool |
|---|---|
| API server | FastAPI |
| Agent orchestration | LangGraph |
| Data storage | PostgreSQL or SQLite |
| Cache/session | Redis |
| Search | Tavily / SerpAPI / browser search API |
| Maps/routing | Google Maps API or OpenRouteService |
| Weather | OpenWeather API |

### AI Components

| Purpose | Tool |
|---|---|
| Main LLM | GPT-5 API or Qwen/Llama/Mistral |
| Fine-tuning | LoRA / QLoRA |
| Fine-tuning library | HuggingFace Transformers + PEFT + TRL |
| Embedding model | sentence-transformers |
| Vector database | ChromaDB or FAISS |

## 7. MVP Scope

The MVP should focus on building a working planning agent, not a full travel booking platform.

### MVP Features

1. User enters destination, trip length, budget, interests, pace, and constraints
2. Preference Agent extracts structured user requirements
3. Research Agent selects candidate places
4. Route Agent groups places into realistic days
5. Budget Agent estimates daily cost
6. Itinerary Agent generates a clean itinerary
7. Critic Agent checks for unrealistic plans
8. User can request a revision

### Example Revision Requests

- “Make it more relaxed.”
- “Remove museums.”
- “Add more food places.”
- “Keep everything under $100 per day.”
- “Avoid long walking.”
- “Make it family friendly.”

## 8. Advanced Features

These are optional if there is extra time:

1. Map view for route groups
2. Weather-aware itinerary changes
3. Budget visualization
4. Memory for user preferences
5. PDF export
6. Hotel area recommendation
7. Comparison between base model and fine-tuned model
8. Evaluation dashboard

## 9. Data Flow

```text
Step 1: User submits request
Step 2: Preference Agent extracts structured fields
Step 3: Missing Information Agent checks required fields
Step 4: Research Agent creates candidate place list
Step 5: Route Agent groups places by day
Step 6: Budget Agent estimates costs
Step 7: Itinerary Agent generates final itinerary
Step 8: Critic Agent reviews feasibility
Step 9: Final answer is returned to frontend
```

## 10. Example Internal State

```json
{
  "destination": "Tokyo",
  "trip_length_days": 5,
  "travelers": "2 friends",
  "budget_level": "medium",
  "interests": ["anime", "food", "shopping", "city views"],
  "pace": "medium",
  "constraints": ["prefer public transit", "avoid expensive restaurants"],
  "selected_places": [
    "Akihabara",
    "Shibuya",
    "Harajuku",
    "Meiji Shrine",
    "Tokyo Skytree",
    "Tsukiji Outer Market"
  ],
  "route_groups": {
    "Day 1": ["Harajuku", "Meiji Shrine", "Shibuya"],
    "Day 2": ["Akihabara", "Tokyo Skytree"],
    "Day 3": ["Tsukiji Outer Market"],
    "Day 4": ["Shinjuku"],
    "Day 5": ["Shibuya"]
  }
}
```

## 11. Evaluation Plan

The project should evaluate whether multi-agent planning improves itinerary quality compared with a single LLM response.

### Systems to Compare

```text
Baseline 1: Single LLM prompt only
Baseline 2: Multi-agent system without fine-tuning
Final System: Multi-agent system + fine-tuned itinerary generator
```

### Evaluation Metrics

| Metric | Meaning |
|---|---|
| Constraint satisfaction | Does the plan follow budget, pace, interests, and constraints? |
| Feasibility | Are activities realistic for each day? |
| Route quality | Are places grouped by nearby locations? |
| Personalization | Does the plan reflect user interests? |
| Format consistency | Is the output valid JSON and consistently structured? |
| Revision quality | Does the system correctly update plans after feedback? |
| Hallucination rate | Does the system invent places or unsupported details? |

## 12. Expected Result

The expected result is that:

1. Single LLM output is fast but less reliable
2. Multi-agent output is more structured and feasible
3. Fine-tuned itinerary generation improves formatting, personalization, and consistency

## 13. Final Deliverables

1. Web application demo
2. Backend API
3. Multi-agent orchestration graph
4. Fine-tuned itinerary generation model or LoRA adapter
5. Fine-tuning dataset
6. Evaluation report
7. Demo examples
8. Final presentation slides

## 14. Project Selling Point

This project is not just a chatbot that writes travel plans. It is a **planning system** that decomposes a complicated travel request into smaller tasks, uses specialized agents, checks feasibility, and uses fine-tuning to produce consistent structured itineraries.

The main contribution is:

> A multi-agent travel planning assistant that combines orchestration, structured planning, feasibility checking, and fine-tuned itinerary generation.
