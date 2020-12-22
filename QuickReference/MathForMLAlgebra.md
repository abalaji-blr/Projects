# Math For ML - Linear Algebra



[TOC]

## Introduction to Linear Algebra

**Algebra** is a branch of mathematics to solve problems with one or more unknown variables.

**Linear Algebra** deals / studies the properties related

* Linear Equations ( Equations of type $$y = mx + c$$, the exponent of the term is 1)
* Vector Spaces ( Geometric vectors aka **vectors** in general, which can be **added or scaled** together)
* Matrices 

The **Linear Algebra** is called **linear** because, when you plot linear equation on a graph, it will form a straight line. The **Linear Algebra** deals with **multiple linear equations**. The Multiple linear equations are represented as **vectors** and **matrices**.

The **non-linear systems** are well approximated by **linear equations**.

## Vector Space

The **Vector Space** is a collection of objects called **vectors**.

The **Vector Spaces** specifies the number of independent directions in the space. When the vectors are added or scaled, the final result vector stays in the same space, then the vector space is called **closed**. It's also called as **closure** property. It's one of the **axiom / assumption / accepted truth used to deduce further** supported by the vectors.

## Basis Vector

**Basis (aka Unit) Vector** - The minmum number of linearly independent vectors that spans the vector Space. In the case 2D, the unit vectors $$\vec{i}$$ and $$\vec{j}$$ span the vector space and they are linear independent.

The dimension of the vector space depends on the number of **basis vectors**.

**Linearly Dependent and Linearly Independent Vectors** - In a set of vectors, if one vector can be defined as the linear combination of other. Then the set of vectors are called **Linearly dependent** vectors. If they can not be represented that way, then the set of vectors are called as **Linearly Independent** vectors. This is central to definition of **Dimension (cardinality of the basis vector).**

**Evaluating the Linear Independence** - Many ways exists including **determinants** of the matrix.

**Linear Map** also known as **Linear Transformation** or **Linear Function**. - A way to map an m-dimensional object to n-dimensional object. For ex: 3D to 2D transformation or 2D to 3D transformation. The relation of two vector spaces can be expressed as **linear map** and they preserve the sum and scalar multiplication.

**Matrices** provides an easy notion to encode linear maps. They are represented interms of rows and columns.

---

## Orthogonal and Orthonormal Vectors

**Orthogonal Vector:** When two vectors are perpendicular, i.e.., angle between them is 90 degress. They are said to be **orthogonal vectors**. The dot product of two vectors is zero.

**Orthonormal Vector:** When two vectors are perpendicular and their magnitude is 1.

---



## Determinants, Eigen Values

**Determinants** are only defined for **square** matrices. i.e., same number of rows and columns.

**Determinant** is a **scalar** value computed from the  **square matrix** and it *encodes certain properties of linear transformation*. It's represented by **det(A) or |A|**. 

Geometrically, it can be viewed as **volume scaling** factor of the linear transformation described by the matrix. The determinant is **positive or negative** according to whether the linear transformation **preserves or reverses** the orientation of a real vector space.

The **determinant** is zero when one of vector overlaps with another. In otherwords, they are **linearly dependent**.

---



**Eigen Values and Eigen Vectors**: The word **eigen** is adopted from Germany and the meaning is **characteristic or proper**. 

Let **A** be a linear transformation, represented by matrix **A**. If there is a vector **X** such that $$ A X = \lambda X$$ , for some scalar $\lambda$, then the $\lambda$ is called the **eigen value** A and **X** is called **Eigen vector**.

The matrix A is a square matrix (say, **mxm**). The X is eigen vector of size **mx1** (column vector). The eigen value, $\lambda$ is of size **mxm** (diagonal matrix).

In the case of multivariate analysis (many features in the dataset), to reduce the dimension, PCA (Principal Component Analysis) is used. PCA in turn uses Eigen decomposition.

---



## Norms, Dot Product, Cross Product

**Norm** : The **norm** of the *vector* in the **vector space** is a real **non negative value** representing the length, size or the magnitude of the vector. It's represented as **|v|**.

Note: Don't confuse with **modulus**. In the case of real numbers, the modulus is the absolute value or the non negative value of the real number. Modulus is nothing to do with **vector space**. Notation is same though.

In the case of Machine Learning, to identify the error (how far the prediction is different from ground truth), the following norms are used: **L1 norm (absolute distance), L2 norm (square root of sum of squared errors, RMSE)**.

**Manhattan Norm (aka L1 norm)**

**Euclidean Norm (aka L2 norm)**



---



**Dot Product** is also called as **scalar product** as it results in a *scalar* value. This can be explained in algebraic form or using geometry. Of course, the geometric form gives visual intution to the concept. 

The Dot product of two vectors for, $\vec{a}$ and $\vec{b}$ is described as $ a . b = |a||b| cos\theta$, where $\theta​$ is the angle between a and b vectors. 

If two vectors are orthogonal, then a.b = 0 as cos 90 = 0.

If two vectors are co-directional (same directions, angle between them is zero), then a.b = |a| |b| cos 0 = |a| |b|.

**The dot product gives the magnitude / length of the resultant vector.**

Note: The dot product provides the length of the resultant vector. Usually, the direction is not mentioned as the resultant vector is in the same direction.

**Scalar Projection** : The scalar projection of $$\vec{a}$$ on to $$\vec{b}$$ is given by $$a_b = |a| cos \theta$$, where $$\theta$$ is angle between a and b vectors.

Thus **dot product** is also called as **projection product**.

---

**Cross Product**: Given two linearly independent vectors a and b, the cross product is a **vector perpendicular** to both a and b.

$$ a \ X \ b = |a| |b| sin\theta$$

When vectors a and b are orthogonal, then $$ a X b = |a| |b| sin(90)$$ => |a| |b| as sin(90 degrees) = 1.

Thus, the magnitude of the resultant **cross product** vector is the **area of the parallelogram** of having  a and b as sides.

---

## Inverse and Invertible Matrix

Inverse : is a noun.

Invertible : is adjective.

For a square matrix A, if the determinant is non-zero, and there exists $A^{-1}$ then $$A^{-1} A = I$$, where **I** is the identity matrix.



---



To Do:

* Euclidean space, Cartesian Coordiante

## References

* [YouTube Channel - MathTheBeautiful - Complete In-Depth Linear Algebra](https://www.youtube.com/watch?v=Fnfh8jNqBlg&list=PLlXfTHzgMRUKXD88IdzS14F4NxAZudSmv&index=1)
* Linear Algebra - 3Blue1Brown