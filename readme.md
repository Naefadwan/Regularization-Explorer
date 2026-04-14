Module 5 Week A — Stretch: Regularization Explorer

Honors Track. This is an Honors Track stretch assignment. Stretch assignments are for learners who have completed all core assignments, are On Track or Advanced, and are attending consistently. If you are behind on core work, focus there first.

Challenge tier prerequisite: None – this stretch builds on base lab work from Week A.
The Challenge

Build a visualization tool that shows how model coefficients change as regularization strength varies. Using the telecom churn dataset from the Week A lab, train LogisticRegression with 20 different values of C (logarithmically spaced from 0.001 to 100), record the coefficient values at each step, and produce a line plot showing each feature’s coefficient trajectory across the regularization spectrum.

Your visualization should make the effect of regularization immediately visible: which features maintain stable coefficients across regularization strengths, which shrink rapidly, and which get driven to zero. Annotate which features get zeroed out first under L1 (Lasso) versus L2 (Ridge) regularization.

This is a professional model interpretation skill. Understanding how regularization reshapes a model’s coefficient landscape helps you diagnose multicollinearity, identify robust predictors, and choose appropriate regularization strategies for production models.
What You’ll Learn

    How L1 and L2 regularization affect coefficient magnitudes differently
    Visualizing the regularization path as a diagnostic tool
    Identifying which features are robust versus sensitive to regularization
    Logarithmic spacing for hyperparameter exploration
    Connecting visual patterns to the bias-variance tradeoff

Outcome

Your deliverable is a standalone repository containing:
Artifact 	Description
Python script or notebook 	Trains LogisticRegression at 20 values of C, records coefficients, and generates the plot
Regularization path plot 	A line plot with C on the x-axis (log scale) and coefficient value on the y-axis, one line per feature
1-paragraph interpretation 	Explains what the plot reveals about the dataset’s features and how L1 and L2 behave differently

Constraints:

    Use numpy.logspace (or equivalent) to generate 20 values of C from 0.001 to 100
    Train separate models for penalty='l1' and penalty='l2' across all 20 C values (use solver='saga' or solver='liblinear' for L1 compatibility)
    The plot should clearly distinguish L1 and L2 trajectories (separate subplots, color coding, or another visual strategy you design)
    Label or annotate the features that reach zero coefficient under L1 – identify which features are eliminated first as regularization strengthens
    Preprocess features consistently (standardize before fitting) so coefficient magnitudes are comparable

Tips

    When C is very small, regularization is very strong – coefficients will be heavily penalized. When C is large, the model approaches unregularized logistic regression.
    L1 produces sparse solutions (some coefficients go to exactly zero). L2 shrinks coefficients toward zero but rarely reaches it. Your plot should make this difference visually obvious.
    If your plot has too many overlapping lines, consider highlighting the top features by coefficient magnitude and dimming the rest.
    Your interpretation paragraph should connect the visualization to a practical recommendation: based on the regularization paths, would you choose L1 or L2 for this dataset, and why?

Submission

    Create a standalone repository for this stretch assignment (not a GitHub Classroom repo – create your own)
    Push your script, plot, and interpretation to the repository
    Paste your repository URL into TalentLMS -> Module 5 -> Stretch 5A-S1
