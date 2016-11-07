module Main where

import           AI.HNN.FF.Network
import           Control.Monad        (foldM, forM_)
import qualified Data.Vector.Storable as V
import           Text.Printf          (printf)

main :: IO ()
main = do
    net     <- createNetwork 25 [10] 3
    trained <- trainWithReport net 10000 20 train
    forM_ train $ \sample -> do
        let out = output trained sigmoid (fst sample)
        printf "::> Output %s\n" (show out)
        printf "::> Expect %s\n" (show $ snd sample)
        putStrLn "==="

trainWithReport :: Network Float -> Int -> Int -> Samples Float -> IO (Network Float)
trainWithReport net' epochs reports samples = do
    let bursts = epochSet epochs reports
    foldM (\net n -> do
                printf "==> Start burst of %d epochs ...\n" n
                let newnet = trainNTimes n 2 sigmoid sigmoid' net samples
                let err = quadError sigmoid newnet samples
                printf "==> Done. Error %f\n" err
                return newnet
          ) net' bursts

epochSet :: Int -> Int -> [Int]
epochSet epochs 0 = [epochs]
epochSet epochs reports =
    let frac = epochs `div` reports
        rest = epochs `mod` reports
    in reports `replicate` frac ++ [rest]

train :: Samples Float
train =
    [ V.fromList
        [ 0, 0, 1, 0, 0
        , 0, 1, 0, 1, 0
        , 0, 1, 1, 1, 0
        , 0, 1, 0, 1, 0
        , 0, 1, 0, 1, 0
        ] --> V.fromList [1, 0, 0]
    , V.fromList
        [ 0, 1, 1, 1, 0
        , 1, 0, 0, 0, 1
        , 1, 1, 1, 1, 1
        , 1, 0, 0, 0, 1
        , 1, 0, 0, 0, 1
        ] --> V.fromList [1, 0, 0]
    , V.fromList
        [ 0, 1, 1, 1, 0
        , 1, 0, 0, 1, 0
        , 1, 1, 1, 1, 0
        , 1, 0, 0, 1, 0
        , 1, 0, 0, 1, 0
        ] --> V.fromList [1, 0, 0]
    , V.fromList
        [ 1, 1, 0, 0, 0
        , 1, 0, 1, 0, 0
        , 1, 1, 0, 0, 0
        , 1, 0, 1, 0, 0
        , 1, 1, 0, 0, 0
        ] --> V.fromList [0, 1, 0]
    , V.fromList
        [ 1, 1, 1, 0, 0
        , 1, 0, 1, 0, 0
        , 1, 1, 0, 0, 0
        , 1, 0, 1, 0, 0
        , 1, 1, 1, 0, 0
        ] --> V.fromList [0, 1, 0]
    , V.fromList
        [ 1, 1, 1, 1, 1
        , 1, 0, 0, 1, 0
        , 1, 1, 1, 0, 0
        , 1, 0, 0, 1, 0
        , 1, 1, 1, 1, 1
        ] --> V.fromList [0, 1, 0]
    , V.fromList
        [ 0, 0, 1, 1, 1
        , 0, 1, 0, 0, 0
        , 1, 0, 0, 0, 0
        , 0, 1, 0, 0, 0
        , 0, 0, 1, 1, 1
        ] --> V.fromList [0, 0, 1]
    , V.fromList
        [ 0, 1, 1, 1, 1
        , 1, 0, 0, 0, 0
        , 1, 0, 0, 0, 0
        , 1, 0, 0, 0, 0
        , 0, 1, 1, 1, 1
        ] --> V.fromList [0, 0, 1]
    , V.fromList
        [ 1, 1, 1, 1, 1
        , 1, 0, 0, 0, 0
        , 1, 0, 0, 0, 0
        , 1, 0, 0, 0, 0
        , 1, 1, 1, 1, 1
        ] --> V.fromList [0, 0, 1]
    ]
