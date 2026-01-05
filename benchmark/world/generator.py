from __future__ import annotations

import itertools
import random
from typing import Dict, List, Tuple

from .schema import (
    Alias,
    CausalEdge,
    CyclicalPattern,
    Entity,
    EntityRef,
    Event,
    EventStep,
    Fact,
    Lesson,
    MemoryPolicy,
    NegativeKnowledge,
    Preference,
    Procedure,
    Task,
    TimeScopedRole,
    TimeWindow,
    ToolMethodMemory,
    VerbatimItem,
    World,
)


class WorldGenerator:
    """
    Generates a small but richly connected latent world.
    Supports all 6 pillars with comprehensive test data.
    """

    def __init__(self, seed: int = 13) -> None:
        self.rng = random.Random(seed)

    def _make_entity(
        self,
        id_suffix: str,
        type_: str,
        aliases: List[Tuple[str, int, int]],
        roles: List[Tuple[str, int, int]],
        persona: str = None,
        category_memberships: List[str] = None,
    ) -> Entity:
        alias_objs = [
            Alias(surface_form=surface, start_time=start, end_time=end or None, confidence=0.8)
            for surface, start, end in aliases
        ]
        role_objs = [
            TimeScopedRole(role=role, start_time=start, end_time=end or None) for role, start, end in roles
        ]
        return Entity(
            id=id_suffix,
            type=type_,  # type: ignore[arg-type]
            aliases=alias_objs,
            attributes=[],
            roles=role_objs,
            persona=persona,
            category_memberships=category_memberships or [],
        )

    def generate_entities(self) -> Dict[str, Entity]:
        entities: Dict[str, Entity] = {}
        # Pillar 1.1: Entity Resolution - Multiple aliases, coreference
        entities["founder"] = self._make_entity(
            "founder",
            "person",
            [
                ("Ash", 0, 8),
                ("Ashvath", 9, None),
                ("the cofounder", 0, None),
                ("the guy who dropped out of Berkeley", 0, None),
            ],
            [("CTO", 0, None)],
            persona="crisp, terse, rolls back fast when needed",
        )
        entities["pm"] = self._make_entity(
            "pm",
            "person",
            [("Mira", 0, None), ("project lead", 12, None), ("the PM", 4, None)],
            [("PM", 4, None), ("IC", 0, 3)],  # Role change over time for 1.2
            persona="likes timelines first, calm communicator",
        )
        # Pillar 1.2: Relationship updates - manager who left
        entities["ex_manager"] = self._make_entity(
            "ex_manager",
            "person",
            [("Jordan", 0, None), ("my old manager", 0, 15)],
            [("Engineering Manager", 0, 15)],  # Left the company
            persona="detail-oriented, thorough reviews",
        )
        entities["project_alpha"] = self._make_entity(
            "project_alpha",
            "project",
            [("Atlas", 0, None), ("memory layer", 5, None), ("the cluster", 8, None)],
            [],
        )
        # Pillar 1.1: Ambiguous references - another cluster
        entities["project_beta"] = self._make_entity(
            "project_beta",
            "project",
            [("Horizon", 0, None), ("the cluster", 0, 7)],  # Ambiguous with project_alpha
            [],
        )
        entities["service_api"] = self._make_entity(
            "service_api",
            "service",
            [("core API", 0, None), ("public endpoint", 5, None)],
            [],
        )
        entities["service_cache"] = self._make_entity(
            "service_cache",
            "service",
            [("Redis cache", 0, None), ("the cache", 0, None)],
            [],
        )
        entities["event_outage"] = self._make_entity(
            "event_outage",
            "event",
            [("the outage", 10, None), ("downtime spike", 10, None), ("February incident", 10, None)],
            [],
        )
        # Pillar 1.3: Type/Category Membership - inferred startup
        entities["nebula_company"] = self._make_entity(
            "nebula_company",
            "organization",
            [("Nebula", 0, None), ("the company", 0, None), ("our startup", 0, None)],
            [],
            category_memberships=["startup", "tech_company"],  # Inferred from context
        )
        # Pillar 3: Migration event for temporal reasoning
        entities["event_migration"] = self._make_entity(
            "event_migration",
            "event",
            [("the migration", 20, None), ("database migration", 20, None)],
            [],
        )
        return entities

    def generate_facts(self, entities: Dict[str, Entity]) -> List[Fact]:
        facts: List[Fact] = []
        window_always = TimeWindow(start_time=0, end_time=None)
        window_old = TimeWindow(start_time=0, end_time=12)
        window_outage = TimeWindow(start_time=10, end_time=14)

        # Basic dependencies for Pillar 2
        facts.append(
            Fact(
                subject_id="project_alpha",
                predicate="depends_on",
                object=EntityRef(entity_id="service_api"),
                validity=window_always,
                source="system",
                confidence=0.9,
            )
        )
        facts.append(
            Fact(
                subject_id="service_api",
                predicate="depends_on",
                object=EntityRef(entity_id="service_cache"),
                validity=window_always,
                source="system",
                confidence=0.85,
            )
        )
        facts.append(
            Fact(
                subject_id="service_api",
                predicate="deployed_in",
                object="us-west",
                validity=window_always,
                source="log",
                confidence=0.85,
            )
        )
        facts.append(
            Fact(
                subject_id="service_cache",
                predicate="deployed_in",
                object="us-west",
                validity=window_always,
                source="log",
                confidence=0.85,
            )
        )
        # Pillar 2.3: Constraint propagation - cache unavailable
        facts.append(
            Fact(
                subject_id="service_cache",
                predicate="status",
                object="unavailable",
                validity=TimeWindow(start_time=25, end_time=27),
                source="log",
                confidence=0.9,
            )
        )
        facts.append(
            Fact(
                subject_id="service_api",
                predicate="deployed_in",
                object="eu-central",
                validity=window_old,
                source="log",
                confidence=0.6,
            )
        )
        facts.append(
            Fact(
                subject_id="event_outage",
                predicate="impact",
                object=EntityRef(entity_id="service_api"),
                validity=window_outage,
                source="log",
                confidence=0.7,
            )
        )
        facts.append(
            Fact(
                subject_id="pm",
                predicate="led",
                object=EntityRef(entity_id="project_alpha"),
                validity=TimeWindow(start_time=9, end_time=None),
                source="user",
                confidence=0.8,
            )
        )
        # Pillar 1.2: Relationship that changed
        facts.append(
            Fact(
                subject_id="founder",
                predicate="reports_to",
                object=EntityRef(entity_id="ex_manager"),
                validity=TimeWindow(start_time=0, end_time=15),
                source="user",
                confidence=0.9,
            )
        )
        # Pillar 2.4: Belief Revision - name change
        facts.append(
            Fact(
                id="name_v1",
                subject_id="user",
                predicate="name",
                object="Alice",
                validity=TimeWindow(start_time=0, end_time=18),
                source="user",
                confidence=0.95,
            )
        )
        facts.append(
            Fact(
                id="name_v2",
                subject_id="user",
                predicate="name",
                object="Bob",
                validity=TimeWindow(start_time=19, end_time=None),
                source="user",
                confidence=0.95,
                supersedes="name_v1",
            )
        )
        # Pillar 2.4: Location change (partial update)
        facts.append(
            Fact(
                id="location_v1",
                subject_id="user",
                predicate="lives_in",
                object="San Francisco",
                validity=TimeWindow(start_time=0, end_time=22),
                source="user",
                confidence=0.9,
            )
        )
        facts.append(
            Fact(
                id="location_v2",
                subject_id="user",
                predicate="lives_in",
                object="Oakland",
                validity=TimeWindow(start_time=23, end_time=None),
                source="user",
                confidence=0.9,
                supersedes="location_v1",
            )
        )
        # Workplace unchanged (to test partial updates)
        facts.append(
            Fact(
                subject_id="user",
                predicate="works_at",
                object=EntityRef(entity_id="nebula_company"),
                validity=window_always,
                source="user",
                confidence=0.95,
            )
        )
        # Pillar 2.2: Quantified reasoning
        facts.append(
            Fact(
                subject_id="nebula_company",
                predicate="requires",
                object="2FA",
                validity=window_always,
                source="system",
                confidence=1.0,
            )
        )
        facts.append(
            Fact(
                subject_id="founder",
                predicate="member_of",
                object=EntityRef(entity_id="nebula_company"),
                validity=window_always,
                source="user",
                confidence=0.95,
            )
        )
        # Pillar 3.1: Temporal ordering - migration events
        facts.append(
            Fact(
                subject_id="event_migration",
                predicate="started_at",
                object="time_20",
                validity=TimeWindow(start_time=20, end_time=None),
                source="log",
                confidence=0.9,
            )
        )
        facts.append(
            Fact(
                subject_id="user",
                predicate="started_using",
                object="Kubernetes",
                validity=TimeWindow(start_time=15, end_time=None),
                source="user",
                confidence=0.8,
            )
        )
        # Pillar 3.3: Causal facts for explanation
        facts.append(
            Fact(
                subject_id="service_api",
                predicate="deployed_on",
                object="Friday night",
                validity=TimeWindow(start_time=10, end_time=11),
                source="log",
                confidence=0.9,
            )
        )
        facts.append(
            Fact(
                subject_id="tests",
                predicate="run_on",
                object="weekdays only",
                validity=window_always,
                source="system",
                confidence=0.95,
            )
        )
        # Attach attributes to entities for quick lookups.
        for fact in facts:
            subject = entities.get(fact.subject_id)
            if subject:
                subject.attributes.append(fact)
        return facts

    def generate_events(self) -> Dict[str, Event]:
        # Main outage event
        outage_steps = [
            EventStep(description="latency spiked", time=10, actor=EntityRef(entity_id="service_api")),
            EventStep(description="rolled back deploy", time=11, actor=EntityRef(entity_id="founder")),
            EventStep(description="traffic shifted to backup", time=12, actor=EntityRef(entity_id="pm")),
            EventStep(description="wrote post-mortem", time=13, actor=EntityRef(entity_id="pm")),
        ]
        outage_causal_links = [
            CausalEdge(source_step=0, target_step=1, relation="caused investigation"),
            CausalEdge(source_step=1, target_step=2, relation="enabled mitigation"),
            CausalEdge(source_step=2, target_step=3, relation="led to documentation"),
        ]
        # Migration event for Pillar 3
        migration_steps = [
            EventStep(description="ran migration scripts", time=20, actor=EntityRef(entity_id="founder")),
            EventStep(description="rebuilt indexes", time=21, actor=EntityRef(entity_id="service_api")),
            EventStep(description="verified data integrity", time=22, actor=EntityRef(entity_id="pm")),
        ]
        migration_causal_links = [
            CausalEdge(source_step=0, target_step=1, relation="triggered rebuild"),
            CausalEdge(source_step=1, target_step=2, relation="required verification"),
        ]
        return {
            "event_outage": Event(
                id="event_outage",
                name="February Outage",
                steps=outage_steps,
                participants=[EntityRef(entity_id="founder"), EntityRef(entity_id="pm")],
                start_time=10,
                end_time=13,
                causal_links=outage_causal_links,
            ),
            "event_migration": Event(
                id="event_migration",
                name="Database Migration",
                steps=migration_steps,
                participants=[EntityRef(entity_id="founder"), EntityRef(entity_id="pm")],
                start_time=20,
                end_time=22,
                causal_links=migration_causal_links,
            ),
        }

    def generate_preferences(self) -> List[Preference]:
        return [
            # Pillar 4.1: Explicit preferences
            Preference(
                subject_id="user",
                signal="prefers Python over Go",
                scope="programming_languages",
                strength=0.9,
                confidence=0.95,
                inferred=False,
                rank=1,
                first_mentioned=0,
                last_mentioned=25,
            ),
            Preference(
                subject_id="founder",
                signal="prefers concise explanations",
                scope="status_updates",
                strength=0.9,
                confidence=0.8,
                inferred=False,
                first_mentioned=5,
                last_mentioned=20,
            ),
            Preference(
                subject_id="pm",
                signal="likes timelines before causes",
                scope="incident_reports",
                strength=0.7,
                confidence=0.75,
                inferred=False,
                first_mentioned=8,
                last_mentioned=15,
            ),
            # Pillar 4.3: Context-dependent preferences (apparent conflict)
            Preference(
                subject_id="user",
                signal="hates meetings in the morning",
                scope="morning_meetings",
                strength=0.8,
                confidence=0.9,
                inferred=False,
                first_mentioned=3,
                last_mentioned=18,
            ),
            Preference(
                subject_id="user",
                signal="loves 1:1 meetings anytime",
                scope="one_on_one_meetings",
                strength=0.85,
                confidence=0.85,
                inferred=False,
                first_mentioned=5,
                last_mentioned=22,
            ),
            # Pillar 4.4: Ranked preferences
            Preference(
                subject_id="user",
                signal="prefers VS Code",
                scope="editors",
                strength=0.9,
                confidence=0.9,
                inferred=False,
                rank=1,
                first_mentioned=2,
                last_mentioned=20,
            ),
            Preference(
                subject_id="user",
                signal="likes Vim as fallback",
                scope="editors",
                strength=0.6,
                confidence=0.8,
                inferred=False,
                rank=2,
                first_mentioned=2,
                last_mentioned=10,
            ),
            Preference(
                subject_id="user",
                signal="tolerates Emacs if necessary",
                scope="editors",
                strength=0.3,
                confidence=0.7,
                inferred=False,
                rank=3,
                first_mentioned=2,
                last_mentioned=5,
            ),
            # Pillar 4.5: Preference drift - React enthusiasm that faded
            Preference(
                subject_id="user",
                signal="loves React",
                scope="frontend_frameworks",
                strength=0.4,  # Reduced from initial high
                confidence=0.6,
                inferred=True,  # Now inferred as lower priority
                first_mentioned=0,
                last_mentioned=8,  # Not mentioned recently
            ),
            # Pillar 4.6: Hard vs soft constraints
            Preference(
                subject_id="user",
                signal="prefers window seats",
                scope="travel",
                strength=0.6,
                confidence=0.8,
                inferred=False,
                is_hard_constraint=False,  # Soft - can be violated
            ),
            Preference(
                subject_id="user",
                signal="allergic to peanuts",
                scope="food",
                strength=1.0,
                confidence=1.0,
                inferred=False,
                is_hard_constraint=True,  # Hard - never violate
            ),
        ]

    def generate_policies(self) -> List[MemoryPolicy]:
        return [
            MemoryPolicy(fact_predicate="deployed_in", decay_half_life=60, retrievable_after_expiry=True),
            MemoryPolicy(fact_predicate="impact", decay_half_life=45, retrievable_after_expiry=False),
            MemoryPolicy(fact_predicate="status", decay_half_life=7, retrievable_after_expiry=True),
        ]

    # === Pillar 1.4: Task/State Tracking ===
    def generate_tasks(self) -> List[Task]:
        return [
            Task(
                id="task_pr_review",
                description="Review the authentication PR",
                assignee=EntityRef(entity_id="founder"),
                status="open",
                created_time=18,
                completed_time=None,
                reminder_time=19,
                commitment_text="You said you'd review the PR tomorrow",
            ),
            Task(
                id="task_auth_refactor",
                description="Finish the auth refactor",
                assignee=EntityRef(entity_id="user"),
                status="completed",
                created_time=10,
                completed_time=16,
            ),
            Task(
                id="task_api_docs",
                description="Write API documentation",
                assignee=EntityRef(entity_id="pm"),
                status="blocked",
                created_time=12,
                completed_time=None,
                blocked_by=["task_api_design"],
            ),
            Task(
                id="task_api_design",
                description="Finalize API design",
                assignee=EntityRef(entity_id="founder"),
                status="in_progress",
                created_time=11,
                completed_time=None,
            ),
            Task(
                id="task_followup_deploy",
                description="Follow up on deployment status",
                assignee=EntityRef(entity_id="user"),
                status="open",
                created_time=20,
                completed_time=None,
                reminder_time=22,
            ),
        ]

    # === Pillar 2.5: Verbatim Recall ===
    def generate_verbatim_items(self) -> List[VerbatimItem]:
        return [
            VerbatimItem(
                id="verbatim_email_regex",
                item_type="code_block",
                content=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                description="regex pattern for email validation",
                created_time=5,
                context="email parsing discussion",
            ),
            VerbatimItem(
                id="verbatim_crash_error",
                item_type="error_message",
                content="FATAL: connection pool exhausted after 30s timeout; max_connections=100",
                description="database connection error from the crash",
                created_time=10,
                context="February outage investigation",
            ),
            VerbatimItem(
                id="verbatim_dashboard_query",
                item_type="query",
                content="SELECT user_id, COUNT(*) as requests FROM api_logs WHERE timestamp > NOW() - INTERVAL '24 hours' GROUP BY user_id ORDER BY requests DESC LIMIT 10",
                description="SQL query for top users dashboard",
                created_time=15,
                context="analytics dashboard setup",
            ),
            VerbatimItem(
                id="verbatim_k8s_config",
                item_type="config",
                content='apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: core-api\nspec:\n  replicas: 3',
                description="Kubernetes deployment config for core API",
                created_time=12,
                context="infrastructure setup",
            ),
        ]

    # === Pillar 3.4: Cyclical Patterns ===
    def generate_cyclical_patterns(self) -> List[CyclicalPattern]:
        return [
            CyclicalPattern(
                id="pattern_code_reviews",
                subject_id="user",
                pattern_type="weekly",
                description="does code reviews on Fridays",
                day_of_week=4,  # Friday
            ),
            CyclicalPattern(
                id="pattern_standup",
                subject_id="user",
                pattern_type="daily",
                description="attends standup in the morning",
                time_of_day="morning",
                exceptions=["never on weekends"],
            ),
            CyclicalPattern(
                id="pattern_running",
                subject_id="user",
                pattern_type="daily",
                description="goes running in the mornings",
                time_of_day="morning",
                exceptions=["never when it rains", "not on Sundays"],
            ),
            CyclicalPattern(
                id="pattern_friday_meeting",
                subject_id="pm",
                pattern_type="weekly",
                description="has sync meeting on Fridays",
                day_of_week=4,
            ),
        ]

    # === Pillar 6.1/6.2: Lessons ===
    def generate_lessons(self) -> List[Lesson]:
        return [
            Lesson(
                id="lesson_migration_order",
                trigger_context="deploying or making database changes",
                lesson_text="Always run migrations before deploy - learned from February outage when deploy failed because migrations didn't run",
                created_time=13,
                source_event="event_outage",
                is_systemic=True,
            ),
            Lesson(
                id="lesson_friday_deploy",
                trigger_context="about to deploy on Friday",
                lesson_text="Avoid Friday deployments - tests don't run on weekends so bugs go unnoticed until Monday",
                created_time=13,
                source_event="event_outage",
                is_systemic=True,
            ),
            Lesson(
                id="lesson_cache_clear",
                trigger_context="experiencing stale data issues",
                lesson_text="Error X (stale reads) is usually caused by cache invalidation lag - clear the cache first",
                created_time=18,
                is_systemic=True,
            ),
            Lesson(
                id="lesson_timeout_oneoff",
                trigger_context="connection timeout",
                lesson_text="Last timeout was due to VPN issues - one-off, not systemic",
                created_time=20,
                is_systemic=False,
            ),
        ]

    # === Pillar 6.3: Procedures ===
    def generate_procedures(self) -> List[Procedure]:
        return [
            Procedure(
                id="proc_dev_setup",
                name="dev environment setup",
                steps=[
                    "Clone the repository",
                    "Run npm install",
                    "Copy .env.example to .env",
                    "Run docker-compose up -d",
                    "Run npm run migrate",
                    "Run npm start",
                ],
                last_updated=15,
            ),
            Procedure(
                id="proc_deploy",
                name="production deployment",
                steps=[
                    "Run all tests locally",
                    "Create a PR and get approval",
                    "Merge to main",
                    "Wait for CI to pass",
                    "Run migrations if needed",
                    "Deploy via kubectl apply",
                    "Monitor metrics for 15 minutes",
                ],
                conditions={
                    "if migrations fail": "rollback immediately and investigate",
                    "if latency spikes": "scale up replicas first, then investigate",
                },
                last_updated=18,
            ),
            Procedure(
                id="proc_incident_response",
                name="incident response",
                steps=[
                    "Acknowledge the alert",
                    "Check dashboards for scope",
                    "Notify the team in #incidents",
                    "Investigate root cause",
                    "Apply fix or rollback",
                    "Write post-mortem within 24 hours",
                ],
                conditions={
                    "if customer-facing": "escalate to on-call lead",
                    "if data loss suspected": "pause all writes immediately",
                },
                last_updated=14,
            ),
        ]

    # === Pillar 6.4: Tool/Method Memory ===
    def generate_tool_memories(self) -> List[ToolMethodMemory]:
        return [
            ToolMethodMemory(
                id="tool_cache_fix",
                problem_context="stale data appearing in the UI",
                solution="clearing the Redis cache fixed it",
                worked=True,
                created_time=16,
                environment="staging",
            ),
            ToolMethodMemory(
                id="tool_restart_fail",
                problem_context="API returning 500 errors",
                solution="restarting the server did NOT help",
                worked=False,
                created_time=10,
                environment="production",
            ),
            ToolMethodMemory(
                id="tool_staging_only",
                problem_context="memory leak in service",
                solution="reducing worker count helps on staging but not production due to load",
                worked=True,
                created_time=19,
                environment="staging",
            ),
            ToolMethodMemory(
                id="tool_index_rebuild",
                problem_context="slow query performance",
                solution="rebuilding the database indexes fixed query times",
                worked=True,
                created_time=21,
                environment="production",
            ),
        ]

    # === Pillar 5.1: Negative Knowledge ===
    def generate_negative_knowledge(self) -> List[NegativeKnowledge]:
        return [
            NegativeKnowledge(
                id="neg_salary",
                topic="salary",
                description="User has never mentioned their salary",
            ),
            NegativeKnowledge(
                id="neg_ski",
                topic="ski plans",
                description="User has never mentioned ski plans or skiing",
            ),
            NegativeKnowledge(
                id="neg_pets",
                topic="pets",
                description="User has not mentioned having any pets",
            ),
            NegativeKnowledge(
                id="neg_spouse",
                topic="spouse or partner",
                description="User has not mentioned a spouse or partner",
            ),
        ]

    def build_world(self) -> World:
        entities = self.generate_entities()
        facts = self.generate_facts(entities)
        events = self.generate_events()
        preferences = self.generate_preferences()
        policies = self.generate_policies()
        tasks = self.generate_tasks()
        verbatim_items = self.generate_verbatim_items()
        cyclical_patterns = self.generate_cyclical_patterns()
        lessons = self.generate_lessons()
        procedures = self.generate_procedures()
        tool_memories = self.generate_tool_memories()
        negative_knowledge = self.generate_negative_knowledge()
        
        return World(
            entities=entities,
            facts=facts,
            events=events,
            preferences=preferences,
            memory_policies=policies,
            tasks=tasks,
            verbatim_items=verbatim_items,
            cyclical_patterns=cyclical_patterns,
            lessons=lessons,
            procedures=procedures,
            tool_memories=tool_memories,
            negative_knowledge=negative_knowledge,
        )


