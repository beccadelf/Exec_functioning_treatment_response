# Exec_functioning_treatment_response

## **The two central scripts of this Repo are "EF_scores_calculation.Rmd" and "Group_Comparison.Rmd".**

### 1. EF_scores_calculation.Rmd performs the pre-processing calculations of the Balanced Integrated Score (BIS) for the three/four (?) cognitive tasks investigated in this study:
**(Each of these tasks is separated for BIS-adequate analysis into its respective conditions)**

   a. _Spatial 2-back Task_: Participants are required to monitor a sequence of spatial stimuli and indicate whether the current stimulus matches the one presented two steps earlier, assessing working memory.
   - **Total**: overall performance
   - **Target**: match between current stimulus and that two steps earlier
   - **Foil**: mismatch between current stimulus and that two steps earlier

   b. _Stroop Task_: Participants must name the color of the font that words are written in, while ignoring the semantic meaning of the word, especially when the color and word are incongruent, to assess cognitive control and attention.
   - **Congruent**: match between font color and semantic meaning
   - **Incongruent**: mismatch between font color and semantic meaning

   c. _Number-Letter Task_: Participants must identify whether a number or letter stimulus is presented based on alternating rules, measuring task-switching and cognitive flexibility.
   - **Repeat**: position of number-letter pair matches between two iterations
   - **Switch**: position of number-letter pair changes between two iterations

   d. _Stop Signal Task_: Participants perform a simple go/no-go task where they must inhibit their response when a stop signal is presented after initiating a response, assessing response inhibition.
   - (Details TBD)

### 2. Group_Comparison.Rmd performs two statistical analyses on this pre-processed data:

   a. **Independent sample Welch's t-test (HC vs. Pat)**: The performance of healthy controls is compared against the performance of the patients for each task condition, respectively.

   b. **Dependent sample Welch's t-test (Pre-Post)**: The performance of patients before the intervention is compared against the performance of patients after the intervention for each task condition.

---

<u><h2><strong>The folder "Exploratory analysis" contains several supplementary analyses that navigate alternative statistical options to arrive at the one ultimately employed.</strong></h2></u>

### 1. Analyses _without_ wrong responses
- We asked ourselves if the removal of incorrect responses by a participant to a task trial would alter the analyses.
- Applications of the BIS pre-processing occasionally adopt a strategy that removes wrong answers (REF) with the rationale...
- However, removing wrong responses resulted in negligible deviations. Hence, we decided for a complete analysis including both correct and incorrect trial responses.

### 2. Testing Normality
- One assumption of t-tests is normality of the analyzed distributions.
- The independent sample t-test assumes normal distribution in the two compared samples, whereas the dependent sample t-test assumes normality in the difference values between the two dependent conditions.
- However, this assumption is frequently neglected (REF), because...
- Since we tested for normal distributions with both statistical and visual analyses, and to follow open science principles, we included these analyses nonetheless.
- Thus, ignoring this assumption could pose a potential limitation to this study.
