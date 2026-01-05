"""
Templated, semi-structured utterance fragments to avoid verbatim leakage.
Organized by pillar for comprehensive benchmark coverage.
"""

TEMPLATES = {
    # === General entity references ===
    "entity_reference": [
        "my cofounder",
        "the person who rolled back the deploy",
        "the PM on Atlas",
        "the project lead",
        "our old service",
        "the public endpoint",
        "the guy who dropped out of Berkeley",
        "my old manager",
    ],
    
    # === Pillar 1: World Modeling ===
    "dependency": [
        "{A} depends on {B}",
        "{B} is required before {A} can launch",
        "{A} will stall unless {B} stays healthy",
        "{A} can't function without {B}",
    ],
    "relationship": [
        "{A} reports to {B}",
        "{A} works with {B}",
        "{A} used to be managed by {B}",
        "{B} left the company, {A} now reports directly",
    ],
    "role_change": [
        "{A} is now leading {B}",
        "{A} took over {B} from {C}",
        "{A} was promoted to {role}",
        "{A} stepped down from {role}",
    ],
    "task_commitment": [
        "I'll review the {task} tomorrow",
        "remind me to {task}",
        "{person} said they'd {task}",
        "need to follow up on {task}",
        "the {task} is blocked by {blocker}",
        "finally finished the {task}",
    ],
    "entity_category": [
        "we're a startup now",
        "Nebula is officially a company",
        "we're fundraising this quarter",
        "we have employees and a product",
    ],
    
    # === Pillar 2: Declarative Reasoning ===
    "fact_chain": [
        "{A} uses {B}, and {B} is deployed in {region}",
        "all {category} require {requirement}",
        "{A} is a {category}, so it must have {requirement}",
    ],
    "belief_revision": [
        "actually my name is {new_name} now",
        "correction: I moved to {new_location}",
        "forget what I said about {topic}",
        "I was wrong about {topic}",
    ],
    "verbatim_share": [
        "here's the regex we used: {content}",
        "the exact error was: {content}",
        "save this SQL query: {content}",
        "our config looks like: {content}",
    ],
    "counterfactual": [
        "if we had used Go, we'd be faster",
        "if only we had more time",
        "had we known, we would have done {alternative}",
    ],
    
    # === Pillar 3: Temporal & Episodic ===
    "incident": [
        "latency climbed and we backed out",
        "we failed over the traffic",
        "downtime spike forced a rollback",
        "the February incident was bad",
    ],
    "temporal_sequence": [
        "first {step1}, then {step2}",
        "{event1} happened before {event2}",
        "after {event1}, we saw {event2}",
        "this was right around when we started using {tool}",
    ],
    "causal_explanation": [
        "because {cause}, we saw {effect}",
        "{effect} was caused by {cause}",
        "the root cause was {cause}",
        "we traced it back to {cause}",
    ],
    "cyclical_pattern": [
        "I do {activity} on {day}s",
        "every {frequency} I {activity}",
        "my usual {day} {activity}",
        "I always {activity} in the {time_of_day}",
        "except when {exception}",
    ],
    "migration_narrative": [
        "during the migration, {event}",
        "the migration started when {event}",
        "after the migration, {result}",
    ],
    
    # === Pillar 4: Preference Learning ===
    "preference": [
        "keep it crisp next time",
        "timeline first, causes later please",
        "short and direct updates only",
        "I prefer {option_a} over {option_b}",
    ],
    "preference_explicit": [
        "I prefer {choice}",
        "I always use {tool}",
        "I like {thing} for {context}",
        "{thing} is my go-to for {context}",
    ],
    "preference_ranked": [
        "{first} is my top choice, then {second}, then {third}",
        "I'd pick {first} over {second} any day",
        "if {first} isn't available, I'd go with {second}",
    ],
    "preference_contextual": [
        "I hate {thing} in the {context}",
        "I love {thing} when {condition}",
        "for {context}, I always prefer {choice}",
    ],
    "hard_constraint": [
        "I'm allergic to {allergen}",
        "I absolutely cannot {constraint}",
        "never {constraint} under any circumstances",
    ],
    "soft_constraint": [
        "I prefer {preference} but it's not a dealbreaker",
        "ideally {preference}, but flexible",
        "if possible, {preference}",
    ],
    
    # === Pillar 5: Knowledge Boundaries ===
    "negative_knowledge_test": [
        "have I ever mentioned {topic}?",
        "do you know my {topic}?",
        "what did I say about {topic}?",
    ],
    "stale_context": [
        "I was sick a while back but I'm fine now",
        "that project ended months ago",
        "we used to do {old_practice} but not anymore",
    ],
    
    # === Pillar 6: Procedural Knowledge ===
    "lesson_learned": [
        "never {bad_practice} again - learned that the hard way",
        "always {good_practice} - that saved us during the outage",
        "the lesson was: {lesson}",
        "from now on, we {practice}",
    ],
    "procedure_step": [
        "to set up dev environment: first {step1}, then {step2}",
        "the process is: {steps}",
        "standard procedure is {steps}",
    ],
    "conditional_procedure": [
        "if you see error {error}, do {fix}",
        "when {condition}, always {action}",
        "if that doesn't work, try {alternative}",
    ],
    "tool_memory": [
        "last time we fixed this by {solution}",
        "we tried {solution} but that didn't help",
        "this fix works on {env} but not {other_env}",
        "clearing the {component} usually fixes this",
    ],
    "proactive_warning": [
        "before you deploy, remember to {precaution}",
        "watch out for {issue} when doing {action}",
        "heads up: {warning}",
    ],
    
    # === Noise (unchanged, for distraction) ===
    "noise": [
        "also I bought a new keyboard",
        "remember to hydrate",
        "random thought about hiking",
        "misc: lunch was great",
        "scheduling dentist next week",
        "thinking about movie night",
        "need to reorder coffee beans",
        "playlist recommendations?",
        "stretch break reminder",
        "weather's been nice lately",
        "coffee machine is broken again",
        "traffic was terrible today",
    ],
}


