# 3ClassLinearClassifier

This program is a basic linear classifier for 3 classes. This model is trained by computing the centroid of each class (e.g. A,B, and C) then constructs a discriminant function between each pair of classes (e.g A/B, B/C, and A/C), halfway between the two centroids and orthogonal to the line connecting the two centroids. For testing it checks the discriminant function between "A or B" and then depending on that answer, decides "A or C" or "B or C". 
