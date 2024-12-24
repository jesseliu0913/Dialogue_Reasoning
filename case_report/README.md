# Clinic_DG

## General Questions:
- **Question**: Describe the patient's personal information.
- **Question**: Describe the patient's experience.
- **Question**: Did you notice any symptoms, such as a fever, cough, or respiratory issues?
- **Question**: What does the imaging (only provide figure explanation here) suggest?
- **Question**: What does the examination suggest?
- **Question**: Are there any suggestions?

---

## Pipeline:
1. **Stage 1**: Use the general questions to retrieve the evidence list from the provided case report.
2. **Stage 2**: Generate questions or responses for one or multiple pieces of evidence and combine them into a dialogue.
3. **Stage 3**: Combine the separate dialogues into one. At this step, all the main information is extracted from the case report.
4. **Final Stage**: Order and polish the dialogue.

---

