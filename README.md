# LoanStatusPrediction
Automate the loan eligibility process based on customer details provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History. 

**Technical design of this project mainly consists of 3 parts as given below**
* Building a Predictive Model
* Evaluate the model.
* Refine the model, as appropriate

**Approch**

This project is designed by selecting decision trees as a prediction model. Decision Tree is one of the most powerful and popular algorithms. Decision-tree algorithm falls under the category of supervised learning algorithms. It works for both continuous as well as categorical output variables.
Decision tree is a flowchart-like tree structure,
Internal node (denoted by rectangles) denotes a test on an attribute,
Each branch represents an outcome of the test, and
Each leaf node (or terminal node, denoted by ovals ) holds a class label.

**Assumptions**

Below are the assumptions made while using Decision tree
At the beginning, consider the whole training set as the root.
Attributes are assumed to be categorical for information gain and for gini index, attributes are assumed to be continuous.
On the basis of attribute values records are distributed recursively.
We use statistical methods for ordering attributes as root or internal node.

**Libraries Used** 
1. Sklearn
2. Pandas
3. Pydotplus
4. IPython

Accuracy of Decision tree created with criteria as Gini ( Default )
Gini Index is a metric to measure how often a randomly chosen element would be incorrectly identified.
It means an attribute with lower gini index should be preferred.
Sklearn supports “gini” criteria for Gini Index and by default, it takes “gini” value.

Accuracy of Decision tree after optimized with criteria as entropy
Entropy is the measure of uncertainty of a random variable, it characterizes the impurity of an arbitrary collection of examples. The higher the entropy the more the information content.
