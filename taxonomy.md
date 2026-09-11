# Intent Taxonomy — Uber Support Agent
**Version:** `1.0.0` (FROZEN)  
**Status:** Frozen following empirical discovery on held-out customer conversations. Post-freeze modifications are prohibited without a formal decision log entry.  
**Coverage Audit:** 95.0% coverage on a 100 fresh held-out sample (`Other_Or_Unclear` rate = 5.0% <= 10.0% threshold).

---

## 1. Overview & Taxonomy Policy

This taxonomy defines **8 mutually exclusive, pairwise distinguishable actionable intents** plus one **`Other_Or_Unclear`** fallback bucket for the `Uber_Support` domain. It was discovered from KMeans clustering on sentence embeddings (`all-MiniLM-L6-v2`) of 2,500 inbound messages and verified against 100 fresh held-out messages.

### Multi-Intent & Ambiguity Resolution Policy
1. **Primary Actionable Intent**: When a customer message touches multiple issues (e.g. complains about a rude driver and demands a refund), annotators and classifiers must assign the **primary actionable ask** — the single core action the support agent must execute to resolve the ticket.
2. **Safety/Conduct Precedence**: If a message involves serious driver misconduct or a physical safety incident, `Driver_Behavior_Or_Safety` takes precedence over financial disputes.
3. **Cancellation Fee Specificity**: If a dispute specifically concerns a cancellation charge, `Cancellation_Fee_Dispute` takes precedence over general `Fare_Dispute_Or_Refund`.

---

## 2. Intent Catalog

### 1. `Fare_Dispute_Or_Refund`
- **Definition**: Customer contests an overcharge, incorrect route fee, unexpected toll, cleaning surcharge, or requests a trip refund.
- **Scope & Boundaries**: Covers financial disputes regarding a completed or attempted trip. Does NOT cover cancellation fees (see `Cancellation_Fee_Dispute`) or general app billing/promo technical errors.
- **Real Examples**:
  1. *"@customer @Uber_Support you charged me FIFTY FIVE DOLLARS for a trip i never took, your service sucks and i want my money back"*
  2. *"Poor service experience company looting customer's money in pool booking ride shown 99 rupees but I was asked to pay 321rupees. Here are proofs for the same [URL]"*
  3. *"@Uber_Support This isn't good enough, I have sent multiple messages proving you charged me incorrectly for the toll but I've had no response."*

---

### 2. `Cancellation_Fee_Dispute`
- **Definition**: Customer disputes a cancellation charge incurred when a driver cancelled, failed to show up, or when the rider cancelled after an excessive wait.
- **Scope & Boundaries**: Specifically isolated to cancellation fees. If a customer is complaining about a general trip fare dispute without mention of a cancellation fee, use `Fare_Dispute_Or_Refund`.
- **Real Examples**:
  1. *"@customer Driver accepted ride, drove opposite direction, then cancelled and I got charged a $5 fee! Refund this now [URL]"*
  2. *"@Uber_Support Why was I charged a cancellation fee when the driver never even showed up at the pickup location?"*
  3. *"@Uber_Support driver called to ask my drop off location and then told me to cancel so I got penalized $5. Please remove the fee."*

---

### 3. `Lost_Item_Inquiry`
- **Definition**: Customer requests assistance to contact a driver or recover personal belongings left behind in an Uber vehicle.
- **Scope & Boundaries**: Covers lost or forgotten items (phones, keys, wallets, luggage). If the customer reports that a driver stole an item or extorted money for its return, classify as `Driver_Behavior_Or_Safety`.
- **Real Examples**:
  1. *"@customer left my phone in an uber last night and driver is not answering. please help me get in contact with him ASAP!"*
  2. *"@Uber_Support I left my wallet in the car this morning. Sent a message through the app 2 hours ago. Need help right away!"*
  3. *"@Uber_Support my friend left her keys in the black Camry we took from downtown. Can you connect us with the driver?"*

---

### 4. `Driver_Behavior_Or_Safety`
- **Definition**: Customer reports unprofessional conduct, dangerous driving, verbal abuse, harassment, route refusal, or safety concerns involving the driver.
- **Scope & Boundaries**: Pertains to driver quality and safety. Triggers the safety escalation workflow if physical threats, accidents, or illegal behavior are described.
- **Real Examples**:
  1. *"@Uber_Support driver was driving dangerously, swerving between lanes and looking at his phone the entire ride. Very unsafe."*
  2. *"@Uber_Support hi this driver cancelled my trip after calling and finding out where I was going... Isn't that against your rules? [URL]"*
  3. *"@Uber_Support i never received a call from the driver and the support person I spoke with was very rude and belittling."*

---

### 5. `Pickup_Or_Arrival_Issue`
- **Definition**: Live logistical issues where the driver cannot locate the rider, arrived at the wrong spot, drove away, or no cars are available.
- **Scope & Boundaries**: Pre-trip arrival and pickup logistics. If the driver fails to arrive and the customer is charged a cancellation fee, classify under `Cancellation_Fee_Dispute`.
- **Real Examples**:
  1. *"What's the deal @Uber_Support? Can't find rides in Canberra at 8:30am? Driver circled 3 times and couldn't find the terminal."*
  2. *"@Uber_Support driver is marked as arrived but there is literally no car on this entire street. Where is he?"*
  3. *"@Uber_Support app GPS sent the driver to the rear alley instead of the main lobby entrance and he won't answer my calls."*

---

### 6. `Account_Access_Or_App_Technical`
- **Definition**: Customer experiences login failure, password reset glitches, app crashes, payment method setup errors, or notification/promo preference issues.
- **Scope & Boundaries**: Digital platform, authentication, and app functionality issues unrelated to a specific physical trip.
- **Real Examples**:
  1. *"@Uber_Support Can’t log into my account. Keeps saying error occurred when trying to reset password. Please help."*
  2. *"@Uber_Support App keeps crashing every time I try to open it after the new update. iOS 11.2."*
  3. *"@Uber_Support what do I have to do to stop getting promo alerts? Unsubscribing doesn’t actually work."*

---

### 7. `Delivery_Or_Food_Issue`
- **Definition**: Inquiries and complaints concerning UberEats food delivery, including missing items, wrong orders, cold food, or unfulfilled orders.
- **Scope & Boundaries**: Specifically partitioned to food delivery orders, separating restaurant/courier issues from rideshare trips.
- **Real Examples**:
  1. *"@customer I'm really disappointed in you for delivering the wrong order. I made a simple order from @customer and now I have someone else's food."*
  2. *"@Uber_Support My food was supposed to be delivered an hour ago. Still not here. Been on hold with your customer support for 30 minutes."*
  3. *"@Uber_Support driver marked the food as delivered but nobody showed up at my porch. Driver is not picking up."*

---

### 8. `Support_Status_Or_Escalation_Request`
- **Definition**: Customer follows up on an unanswered ticket, complains about repetitive canned replies, or demands managerial/legal escalation.
- **Scope & Boundaries**: Customer's core intent is addressing delayed or defective support rather than reporting a fresh first-time problem.
- **Real Examples**:
  1. *"@Uber_Support Absolutely horrible customer service. No phone number to call, app tickets ignored for 2 days. Unacceptable."*
  2. *"@Uber_Support I have received responses from all the times I’ve reached out. However, my issue has not been addressed or resolved."*
  3. *"@Uber_Support I am dissatisfied with customer service representative repeating the same thing again and again. Is there no way of escalation?"*

---

### 9. `Other_Or_Unclear`
- **Definition**: Vague statements, greetings with no actionable context, positive feedback, jokes, or out-of-scope inquiries.
- **Scope & Boundaries**: Catch-all bucket for queries that cannot be reliably mapped to the 8 actionable categories above.
- **Real Examples**:
  1. *"@Uber_Support hello good morning"*
  2. *"@Uber_Support shoutout to your driver Dave in Chicago for the awesome playlist!"*
  3. *"@customer @customer lol it's still not working"*

---

## 3. Pairwise Distinguishability Matrix

| Intent A | Intent B | Boundary Distinction Rule |
|---|---|---|
| `Fare_Dispute_Or_Refund` | `Cancellation_Fee_Dispute` | If fee was triggered by a cancellation event, label `Cancellation_Fee_Dispute`; all other ride charges/surcharges belong in `Fare_Dispute_Or_Refund`. |
| `Driver_Behavior_Or_Safety` | `Pickup_Or_Arrival_Issue` | If driver acted aggressively, rudely, or refused destination, label `Driver_Behavior_Or_Safety`; if driver was simply lost or app GPS was inaccurate, label `Pickup_Or_Arrival_Issue`. |
| `Driver_Behavior_Or_Safety` | `Lost_Item_Inquiry` | If rider is seeking to recover forgotten property, label `Lost_Item_Inquiry`, unless driver is threatening or extorting the passenger. |
| `Fare_Dispute_Or_Refund` | `Delivery_Or_Food_Issue` | Food/order item disputes belong in `Delivery_Or_Food_Issue`; passenger vehicle trip fares belong in `Fare_Dispute_Or_Refund`. |
| `Support_Status_Or_Escalation_Request` | Any other intent | If the primary complaint is that a previous ticket was ignored/unresolved, label `Support_Status_Or_Escalation_Request`. If describing a new incident for the first time, label the specific issue. |
