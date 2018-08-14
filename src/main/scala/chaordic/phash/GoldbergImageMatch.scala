package chaordic.phash

import java.awt.image.{BufferedImage, IndexColorModel}
import java.io.{ByteArrayInputStream, InputStream}
import java.security.InvalidParameterException
import java.util

import breeze.linalg.Options.{Dimensions1, Dimensions2, Value}
import breeze.stats.DescriptiveStats
import breeze.storage.Zero
import javax.imageio.ImageIO

import scala.reflect.ClassTag

object GoldbergImageMatch {
  import breeze.linalg._
  import breeze.numerics._
  import breeze.stats._


  object ImageSignature {
    case class Window(top: Int, bottom: Int, left: Int, right: Int)

    def toRGB(data: Array[Byte]): DenseMatrix[Int] = {
      val buffer = ImageIO.read(new ByteArrayInputStream(data))

      val width = buffer.getWidth
      val height = buffer.getHeight

      val target = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
      val graphics = target.createGraphics()
      graphics.drawImage(buffer, 0, 0, width, height, null)
      graphics.dispose()
      val argb = target.getRaster
        .getDataElements(0, 0, width, height, null)
        .asInstanceOf[Array[Int]]
      DenseMatrix(argb).reshape(width, height).t
    }

    def preprocessImage(data: Array[Byte]): DenseMatrix[Double] = {
      val rgb = toRGB(data)
      rgb.mapValues(argbToGrayScale)
    }

    private def argbToGrayScale(argb: Int): Double = {
      val red = ((argb >> 16) & 0xff)
      val green = ((argb >> 8) & 0xff)
      val blue = (argb & 0xff)
      val normalization = 255.0
      // Like https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L785
      (0.2125 * red + 0.7154 * green + 0.0721 * blue) / normalization
    }

    def diff(m: DenseMatrix[Double], axis: Axis = Axis._1): DenseMatrix[Double] =
      axis match {
        case Axis._0 =>
          m(1 to -1, ::) - m(0 to -2, ::)
        case Axis._1 =>
          m(::, 1 to -1) - m(::, 0 to -2)
      }

    def diff(v: DenseVector[Double]): DenseVector[Double] =
      diff(v.toDenseMatrix).toDenseVector

    def kDiag[V:ClassTag:Zero](m: DenseMatrix[V], k: Int): DenseVector[V] = {
      require(m.rows == m.cols, "Matrix must be square")
      require(k >= 0 && k <= m.rows || k < 0 && -k <= m.rows)
      val size = m.rows - Math.abs(k)
      val v = DenseVector.zeros[V](size)
      (0 until size).foreach { i =>
        val x = if (k >= 0) i else i - k
        val y = if (k >= 0) i + k else i
        v(i) = m(x, y)
      }
      v
    }

    def kDiag[V:ClassTag:Zero](vector: DenseVector[V], k: Int): DenseMatrix[V] = {
      val size = vector.size + Math.abs(k)
      val m = DenseMatrix.zeros[V](size, size)
      vector.foreachPair { case (i, v) =>
        val x = if (k >= 0) i else i - k
        val y = if (k >= 0) i + k else i
        m(x, y) = v
      }
      m
    }

    def padAfter[T:ClassTag:Zero](m: DenseMatrix[T], n: Int)(implicit canPad: CanPadRight[DenseMatrix[T], Dimensions2, DenseMatrix[T]]): DenseMatrix[T] = {
      padRight(m, Dimensions2(m.rows + n, m.cols + n), Value(implicitly[Zero[T]].zero))
    }

    def padAfter[T:ClassTag:Zero](m: DenseVector[T], n: Int)(implicit canPad: CanPadRight[DenseVector[T], Dimensions1, DenseVector[T]]): DenseVector[T] = {
      padRight(m, Dimensions1(m.length + n), Value(implicitly[Zero[T]].zero))
    }

    def prepend(vector: DenseVector[Double], v: Double): DenseVector[Double] =
      DenseVector.vertcat(DenseVector(v), vector)

    def searchSorted(m: Array[Double], v: Double, right: Boolean): Int = {
      val result = java.util.Arrays.binarySearch(m, v)
      if (result >= 0) {
        if (right) {
          var finalResult = result
          while (finalResult < m.length - 1 && m(finalResult) == v) {
            finalResult += 1
          }
          finalResult
        } else {
          var finalResult = result
          while (finalResult > 0 && m(finalResult - 1) == v) {
            finalResult -= 1
          }
          finalResult
        }
      } else {
        -(result + 1)
      }
    }

    def cropImage(image: DenseMatrix[Double], cropPercentiles: (Int, Int)) = {
      val (lower, upper) = (cropPercentiles._1 / 100.0, cropPercentiles._2 / 100.0)
      val rw = accumulate(sum(abs(diff(image, Axis._1)), Axis._1)).toArray
      val cw = accumulate(sum(abs(diff(image, Axis._0)), Axis._0)).inner.toArray

      var upperColumnLimit = searchSorted(cw, DescriptiveStats.percentile(cw, upper), right = false)
      var lowerColumnLimit = searchSorted(cw, DescriptiveStats.percentile(cw, lower), right = true)
      var upperRowLimit = searchSorted(rw, DescriptiveStats.percentile(rw, upper), right = false)
      var lowerRowLimit = searchSorted(rw, DescriptiveStats.percentile(rw, lower), right = true)

      if (lowerRowLimit > upperRowLimit) {
        lowerRowLimit = (lower * image.rows).toInt
        upperRowLimit = (upper * image.rows).toInt
      }

      if (lowerColumnLimit > upperColumnLimit) {
        lowerColumnLimit = (lower * image.cols).toInt
        upperColumnLimit = (upper * image.cols).toInt
      }

      Window(lowerRowLimit, upperRowLimit, lowerColumnLimit, upperColumnLimit)
    }

    def computeGridPoints(image: DenseMatrix[Double],
                          n: Int,
                          window: Window
                         ) = {
      val xCoords = linspace(window.top, window.bottom, n + 2)(1 to -2).map(_.toInt)
      val yCoords = linspace(window.left, window.right, n + 2)(1 to -2).map(_.toInt)

      (xCoords, yCoords)
    }

    def computeMeanLevel(image: DenseMatrix[Double],
                         xCoords: DenseVector[Int],
                         yCoords: DenseVector[Int]
                        ) = {

      val P = Math.max(2.0, Math.floor(0.5 + Math.min(image.cols, image.rows) / 20.0)) // from paper

      val avgGrey = DenseMatrix.zeros[Double](xCoords.length, yCoords.length)

      xCoords.foreachPair { case (i, x) =>
        val lowerXlim = Math.max(x - P / 2, 0).toInt
        val upperXlim = Math.min(lowerXlim + P, image.rows).toInt

        yCoords.foreachPair { case (j, y) =>
          val lowerYlim = Math.max(y - P / 2, 0).toInt
          val upperYlim = Math.min(lowerYlim + P, image.cols).toInt
          avgGrey(i, j) = mean(image(lowerXlim until upperXlim, lowerYlim until upperYlim))
        }
      }

      avgGrey
    }

    def computeDifferentials(avgGrey: DenseMatrix[Double], diagonalNeighbors: Boolean) = {

      val rightNeighbors = -DenseMatrix.horzcat(
        diff(avgGrey),
        DenseMatrix.zeros[Double](avgGrey.rows, 1)
      )

      val leftNeighbors = -DenseMatrix.horzcat(
        rightNeighbors(::, -1 to -1),
        rightNeighbors(::, 0 to -2)
      )

      val downNeighbors = -DenseMatrix.vertcat(
        diff(avgGrey, axis = Axis._0),
        DenseMatrix.zeros[Double](1, avgGrey.cols)
      )

      val upNeighbors = -DenseMatrix.vertcat(
        downNeighbors(-1 to -1, ::),
        downNeighbors(0 to -2, ::)
      )

      if (diagonalNeighbors) {
        // this implementation will only work for a square (m x m) grid
        val diagonals = (-avgGrey.rows + 1) until avgGrey.rows

        val upperLeftNeighbors = sum(diagonals.map { i =>
          kDiag(prepend(diff(kDiag(avgGrey, i)), 0), i)
        })
        val lowerRightNeighbors =
          -padAfter(upperLeftNeighbors(1 to -1, 1 to -1), 1)

        val flipped = fliplr(avgGrey)

        val upperRightNeighbors = sum(diagonals.map { i =>
          kDiag(prepend(diff(kDiag(flipped, i)), 0), i)
        })

        val lowerLeftNeighbors =
          -padAfter(upperRightNeighbors(1 to -1, 1 to -1), 1)

        List(
          upperLeftNeighbors,
          upNeighbors,
          fliplr(upperRightNeighbors),
          leftNeighbors,
          rightNeighbors,
          fliplr(lowerLeftNeighbors),
          downNeighbors,
          lowerRightNeighbors
        )
      } else {
        List(
          upNeighbors,
          leftNeighbors,
          rightNeighbors,
          downNeighbors
        )
      }
    }

    def normalizeAndThreshold(diffMat: List[DenseMatrix[Double]],
                              identicalTolerance: Double,
                              nLevels: Int
                             ): Unit = {

      diffMat.foreach { m =>
        m.foreachPair { case (k, v) =>
          if (Math.abs(v) < identicalTolerance)
            m(k) = 0
        }
      }

      if (!diffMat.forall(_.forall(_ == 0.0D))) {
        val flat = diffMat.flatMap(_.toArray)
        val percentiles = linspace(0, 100, nLevels+1).mapValues(_ / 100.0)
        val positiveCutoffs = linspace(0, 100, nLevels+1).map(p => DescriptiveStats.percentile(flat.filter(_ > 0), p / 100.0))
        val negativeCutoffs = linspace(100, 0, nLevels+1).map(p => DescriptiveStats.percentile(flat.filter(_ < 0), p / 100.0))

        (0 until positiveCutoffs.length - 1)
          .map(i => (positiveCutoffs(i), positiveCutoffs(i+1)))
          .zipWithIndex
          .foreach {
            case ((low, high), level) =>
              diffMat.foreach { m =>
                m.foreachPair { case (k, v) =>
                  if (v >= low && v <= high)
                    m(k) = level + 1
                }
              }
          }

        (0 until negativeCutoffs.length - 1)
          .map(i => (negativeCutoffs(i), negativeCutoffs(i+1)))
          .zipWithIndex
          .foreach {
            case ((left, right), level) =>
              diffMat.foreach { m =>
                m.foreachPair { case (k, v) =>
                  if (v <= left && v >= right)
                    m(k) = -(level + 1)
                }
              }
          }
      }
    }

    def normalizedDistance(a: DenseVector[Int], b: DenseVector[Int]): Double = {
      val normDiff = norm(a - b)
      val norm1 = norm(b)
      val norm2 = norm(a)
      normDiff / (norm1 + norm2)
    }

    def getWords(array: DenseVector[Int], k: Int, N: Int): DenseMatrix[Int] = {
      val wordPositions = linspace(0, array.length, N)(0 to -2).map(_.toInt)
      require(k <= array.length, "Word length cannot be longer than array length")
      require(wordPositions.length <= array.length, "Number of words cannot be more than array length")
      val words = DenseMatrix.zeros[Int](N, k)

      wordPositions.foreachPair { case (i, pos) =>
        if (pos + k <= array.length)
          words(i, ::) := array(pos until pos + k).t
        else
          words(i, ::) := padRight(array(pos to -1), k).t
      }
      words
    }

    def maxContrast(words: DenseMatrix[Int]): Unit = {
      words.foreachPair { case (k, v) =>
        if (v > 0)
          words(k) = 1
        else if (v < 0)
          words(k) = -1
      }
    }

    def wordsToInt(words: DenseMatrix[Int]): DenseVector[Int] = {
      // Three states (-1, 0, 1)
      val codingVector = DenseVector((0 until words.cols).map(i => Math.pow(3, i).toInt).toArray).toDenseMatrix
      // The 'plus one' here makes all digits positive, so that the
      // integer representation is strictly non-negative and unique
      val wordsPlus: DenseMatrix[Int] = words + 1
      (wordsPlus * codingVector.t).toDenseVector
    }

  }

  case class ImageSignature(n: Int = 9,
                            cropPercentiles: (Int, Int) = (5, 95),
                            diagonalNeighbors: Boolean = true,
                            identicalTolerance: Double = 2/255.0,
                            nLevels: Int = 2
                           ) {
    import ImageSignature._
    def generateSignature(data: Array[Byte]): DenseVector[Int] = {
      val image = preprocessImage(data)
      val imageLimits = cropImage(image, cropPercentiles)
      val (xCoords, yCoords) = computeGridPoints(image, n = n, window = imageLimits)
      val avgGrey = computeMeanLevel(image, xCoords, yCoords)

      val diffMat = computeDifferentials(avgGrey, diagonalNeighbors)
      normalizeAndThreshold(diffMat, identicalTolerance, nLevels)
      DenseVector(diffMat.flatMap(_.toArray).map(_.toInt).toArray)
    }

    def generateAll(data: Array[Byte], k: Int = 16, N: Int = 63): (DenseVector[Int], DenseVector[Int]) = {
      val sig = generateSignature(data)
      val words = getWords(sig, k, N)
      maxContrast(words)
      val intWords = wordsToInt(words)
      sig -> intWords
    }
  }
}
