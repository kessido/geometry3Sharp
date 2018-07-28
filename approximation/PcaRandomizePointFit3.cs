using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace g3.approximation
{
    /// <summary>
    /// PCA and Randomized strategy for fitting a smaller
    /// oriented bounding box for a set of 3D points.
    /// </summary>
    public partial class PcaRandomizePointFit3
    {
        /// <summary>
        ///     PCA algorithm uses 3 pass over the points.
        ///          1. Find the center of the points.
        ///          2. Find the covariance matrix of the normlized points.
        ///          3. Find the enclosing box of the points.
        ///     The Pca-RandomizedAlgorithm uses the fact that at each pass
        ///     the points are in our cache, and the 3 pass neccessity, 
        ///     and uses it more throughly using gradient descent huristic.
        ///
        ///     At the first step:
        ///             Generate directions randomly.
        ///             Each direction has 3 other "neightbors" direction
        ///             for the gradient decent calculation which
        ///             are used to calculate the derivative.
        ///     At the second\third pass:
        ///             Iterate the gradient decent with
        ///             small step to get close to the minimum.
        ///
        ///     Complexity: O(n)
        /// </summary>
        /// <author>Ido Kessler</author>
        private const int NumberOfPointsToGenerate = 100;

        public static Box3d Fit(IEnumerable<Vector3d> points)
        {
            var randomBoxes = GenerateRandomOrientedBoundingBoxGradientDescent();
            var vector3Ds = points.ToList();
            var center = Vector3d.Zero;
            var numOfPoint = 0;

            // Compute center;
            foreach (var point in vector3Ds)
            {
                randomBoxes.ForEach(o => o.HandlePoint(point));
                center += point;
                ++numOfPoint;
            }
            center /= numOfPoint;

            // Get best box till now.
            var bestBox3D = randomBoxes.Select(o => o.GetMinBox()).MinBox();

            // Gradient Decsent Step.
            randomBoxes.ForEach(o => o.Step());

            // Compute covariance
            var covarianceMatrixComputer = new PcaRotationIterativeComputation(center);
            foreach (var point in vector3Ds)
            {
                randomBoxes.ForEach(o => o.HandlePoint(point));
                covarianceMatrixComputer.AddPoint(point);
            }

            // Compute the PCA
            var pcaRotationMatrix = covarianceMatrixComputer.GetMatrix();

            // Get best box till now.
            bestBox3D = randomBoxes.Select(o => o.GetMinBox()).Concat(new[] { bestBox3D }).MinBox();

            // Gradient Decsent Step.
            randomBoxes.ForEach(o => o.Step());

            var pcaBoundingBox = new IncreasingOrientedBoundingBox(pcaRotationMatrix);
            foreach (var point in vector3Ds)
            {
                randomBoxes.ForEach(o => o.HandlePoint(point));
                pcaBoundingBox.AddPoint(point);
            }

            // Get best box till now.
            bestBox3D = randomBoxes.Select(o => o.GetMinBox()).
                Concat(new[] { bestBox3D, pcaBoundingBox.ToBox3D() }).MinBox();

            return bestBox3D;
        }

        private struct PcaRotationIterativeComputation
        {
            private double _sumXX, _sumXY, _sumXZ, _sumYY, _sumYZ, _sumZZ;
            private int _numberOfPoint;
            private readonly Vector3d _center;

            public PcaRotationIterativeComputation(Vector3d center)
            {
                _center = center;
                _sumXX = 0; _sumXY = 0; _sumXZ = 0; _sumYY = 0; _sumYZ = 0; _sumZZ = 0;
                _numberOfPoint = 0;
            }


            public void AddPoint(Vector3d point)
            {
                point -= _center;
                ++_numberOfPoint;
                _sumXX += point[0] * point[0];
                _sumXY += point[0] * point[1];
                _sumXZ += point[0] * point[2];
                _sumYY += point[1] * point[1];
                _sumYZ += point[1] * point[2];
                _sumZZ += point[2] * point[2];
            }

            public Matrix3d GetMatrix()
            {
                var matrix = new double[] {
                   _sumXX, _sumXY, _sumXZ,
                   _sumXY, _sumYY, _sumYZ,
                   _sumXZ, _sumYZ, _sumZZ
                };
                for (var i = 0; i < matrix.Length; ++i) matrix[i] /= _numberOfPoint;
                var solver = new SymmetricEigenSolver(3, 4096);
                var iters = solver.Solve(matrix, SymmetricEigenSolver.SortType.Increasing);
                var resultValid = (iters > 0 && iters < SymmetricEigenSolver.NO_CONVERGENCE);
                if (resultValid)
                {
                    var evectors = solver.GetEigenvectors();
                    var axisX = new Vector3d(evectors[0], evectors[1], evectors[2]);
                    var axisY = new Vector3d(evectors[3], evectors[4], evectors[5]);
                    var axisZ = new Vector3d(evectors[6], evectors[7], evectors[8]);
                    return new Matrix3d(axisX, axisY, axisZ, true);
                }
                else
                {
                    return Matrix3d.Identity;
                }
            }
        }

        private static List<OrientedBoundingBoxGradientDescent> GenerateRandomOrientedBoundingBoxGradientDescent()
        {
            var res = new List<OrientedBoundingBoxGradientDescent>();
            for (var i = 0; i < NumberOfPointsToGenerate; ++i)
                res.Add(OrientedBoundingBoxGradientDescent.GetRandom());
            return res;
        }

        private class OrientedBoundingBoxGradientDescent
        {
            private const double GradientDescentStepSize = Math.PI * 1 / NumberOfPointsToGenerate;
            private const double GradientDescentDerivativeDiff = 1e-3;

            private Vector3d _rotation;
            private IncreasingOrientedBoundingBox _mainBox, _xDerivativeBox, _yDerivativeBox, _zDerivativeBox;

            public Box3d GetMinBox() => GetBoxes().MinBox();

            public void HandlePoint(Vector3d point)
            {
                _mainBox.AddPoint(point);
                _xDerivativeBox.AddPoint(point);
                _yDerivativeBox.AddPoint(point);
                _zDerivativeBox.AddPoint(point);
            }
            public void Step()
            {
                var currentVolume = _mainBox.Volume;
                var diff = new Vector3d
                {
                    [0] = currentVolume - _xDerivativeBox.Volume,
                    [1] = currentVolume - _yDerivativeBox.Volume,
                    [2] = currentVolume - _zDerivativeBox.Volume
                };
                diff.Normalize();
                _rotation += diff * GradientDescentStepSize;
                GenerateBoxes();
            }

            private static readonly Random Random = new Random();
            public static OrientedBoundingBoxGradientDescent GetRandom()
            {
                return new OrientedBoundingBoxGradientDescent(new Vector3d(Random.NextDouble(), Random.NextDouble(), Random.NextDouble()) * Math.PI * 2);
            }

            private OrientedBoundingBoxGradientDescent(Vector3d rotation)
            {
                _rotation = rotation;
                GenerateBoxes();
            }


            private void GenerateBoxes()
            {
                _mainBox = GenerateBox(_rotation[0], _rotation[1], _rotation[2]);
                _xDerivativeBox = GenerateBox(_rotation[0] + GradientDescentDerivativeDiff, _rotation[1], _rotation[2]);
                _yDerivativeBox = GenerateBox(_rotation[0], _rotation[1] + GradientDescentDerivativeDiff, _rotation[2]);
                _zDerivativeBox = GenerateBox(_rotation[0], _rotation[1], _rotation[2] + GradientDescentDerivativeDiff);
            }



            private static IncreasingOrientedBoundingBox GenerateBox(double xRotation, double yRotation, double zRotation)
            {
                var rotationMatrix = RotationMatrixGenerator.GetRotationMatrix(0, xRotation) *
                                     RotationMatrixGenerator.GetRotationMatrix(1, yRotation) *
                                     RotationMatrixGenerator.GetRotationMatrix(2, zRotation);
                return new IncreasingOrientedBoundingBox(rotationMatrix);
            }


            private IEnumerable<Box3d> GetBoxes()
            {
                yield return _mainBox.ToBox3D();
                yield return _xDerivativeBox.ToBox3D();
                yield return _yDerivativeBox.ToBox3D();
                yield return _zDerivativeBox.ToBox3D();
            }


        }
        private struct IncreasingOrientedBoundingBox
        {
            private readonly Matrix3d _rotationMatrix;
            private Vector3d _min, _max;

            public IncreasingOrientedBoundingBox(Matrix3d rotationMatrix)
            {
                _rotationMatrix = rotationMatrix;
                _min = Vector3d.MaxValue;
                _max = Vector3d.MinValue;
            }

            public void AddPoint(Vector3d point)
            {
                for (var i = 0; i < 3; ++i)
                {
                    _min[i] = Math.Min(_min[i], point[i]);
                    _max[i] = Math.Max(_max[i], point[i]);
                }
            }

            public Box3d ToBox3D() => new Box3d((_min + _max) / 2, _rotationMatrix.Row0, _rotationMatrix.Row1, _rotationMatrix.Row2, (_max - _min) / 2);

            public double Volume
            {
                get { var myVolume = (_max - _min); return myVolume.x * myVolume.y * myVolume.z; }
            }
        }
    }


    internal static class Box3DExtension
    {
        public static Box3d MinBox(this IEnumerable<Box3d> boxes)
        {
            var enumerator = boxes.GetEnumerator();
            var minBox = enumerator.Current;
            var minVolume = minBox.Volume;
            while (enumerator.MoveNext())
            {
                var volume = enumerator.Current.Volume;
                if (volume >= minVolume) continue;
                minVolume = volume;
                minBox = enumerator.Current;
            }
            enumerator.Dispose();
            return minBox;
        }
    }

    internal static class RotationMatrixGenerator
    {
        public static Matrix3d GetRotationMatrix(int axis, double rotation)
        {
            var cos = Math.Cos(rotation);
            var sin = Math.Sin(rotation);
            var res = new Matrix3d();
            var currentOptionIndex = 0;
            double[] options = { cos, -sin, sin, cos };
            for (var r = 0; r < 3; ++r)
                for (var c = 0; c < 3; ++c)
                    res[r, c] = (r == axis || c == axis) ? 0 : options[currentOptionIndex++];
            res[axis, axis] = 1;
            return res;
        }
    }
}