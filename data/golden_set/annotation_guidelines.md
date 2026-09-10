# Golden Set Annotation Guidelines — Uber Support Agent
**Version:** `1.0.0` (FROZEN)  
**Target Evaluation Set Size:** 200 examples  
**Source Pool:** `held_out_eval_pool` (strictly disjoint from retrieval corpus and training data; zero tweet ID and zero text overlap).

---

## 1. Objective

The golden evaluation set establishes the uncompromised ground truth for measuring:
1. **Intent Classification Accuracy & Macro-F1** (across the 8+1 frozen taxonomy in `taxonomy.md`).
2. **Reply Quality & Policy Grounding** (actionable resolution guidance vs. fabricated promises).
3. **Escalation Routing Safety** (precision, recall, and false negative rate on cases requiring human escalation).

---

## 2. Annotation Schema

Each golden record in `golden_set.jsonl` contains:

```json
{
  "example_id": "gold_001",
  "customer_tweet_id": 123456,
  "customer_text": "...",
  "turn_position": "first_turn | follow_up",
  "length_bucket": "short | medium | long",
  "edge_case_category": "standard | ambiguous_multi_intent | sarcasm_implicit_dissatisfaction | safety_legal_escalate | low_info_vague | out_of_distribution",
  "gold_intent": "Fare_Dispute_Or_Refund | Cancellation_Fee_Dispute | Lost_Item_Inquiry | Driver_Behavior_Or_Safety | Pickup_Or_Arrival_Issue | Account_Access_Or_App_Technical | Delivery_Or_Food_Issue | Support_Status_Or_Escalation_Request | Other_Or_Unclear",
  "gold_routing_decision": "AUTO_HANDLE | ESCALATE",
  "gold_escalate_reason": "Clear explanation for the routing decision",
  "gold_reply_reference": "Ideal resolution guidance and instructions for reply generator",
  "brand_historical_reply": "Actual reply from Uber_Support (for reference/baselines)"
}
```

---

## 3. Escalation Rules (`gold_routing_decision`)

### Mandated Escalation Triggers (`ESCALATE`)
A query **MUST** be labelled `ESCALATE` if any of the following criteria are met:
1. **Physical Safety & Severe Misconduct**: The message reports dangerous driving, physical aggression, harassment, vehicle accidents, or criminal behavior (`Driver_Behavior_Or_Safety` with safety risk).
2. **Legal & Regulatory Threats**: Customer explicitly threatens litigation, regulatory complaint, police involvement, or media exposure.
3. **Repeated Unresolved Support Loop**: Customer indicates that prior support attempts have failed or tickets have been ignored (`Support_Status_Or_Escalation_Request`).
4. **Out-of-Distribution / High Financial Impact**: Issues involving severe financial disputes requiring human discretion beyond standard automated self-service refund policies.

### Automated Handling Criteria (`AUTO_HANDLE`)
A query should be labelled `AUTO_HANDLE` only if:
1. It is a standard, self-service actionable issue where proven historical troubleshooting instructions exist (e.g., standard trip fare review instructions, lost item driver contact steps in-app, cancellation fee dispute forms, app password reset guides).
2. The user is not in acute distress or threatening escalation.
3. The query does not require real-time human authority to issue custom financial exceptions.

---

## 4. Edge Case Sampling Categories

To prevent artificially optimistic evaluation scores, the golden set deliberately includes at least **20-25% edge cases**:
1. `ambiguous_multi_intent`: Messages combining multiple distinct problems (e.g., driver arrived late and was rude).
2. `sarcasm_implicit_dissatisfaction`: Bitter or ironic customer messages without overt complaint keywords (e.g., *"Love paying $40 to sit in traffic while driver watches videos on his dash mount"*).
3. `safety_legal_escalate`: Critical safety, police, accident, or legal threat reports.
4. `low_info_vague`: Minimalist inquiries with insufficient details (e.g., *"help"*, *"this app is broken"*).
5. `out_of_distribution`: Questions regarding unsupported services, city policy changes, or services outside standard Uber rideshare/Eats scope.
