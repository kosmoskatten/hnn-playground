module Mnist
    ( mnistFromFile
    ) where

import           AI.HNN.FF.Network
import qualified Data.ByteString.Lazy.Char8 as LBS
import           Data.Csv
import qualified Data.Vector                as V
import qualified Data.Vector.Storable       as VS

-- | Dummy wrapper ...
newtype Bundle = Bundle (Sample Float)

instance FromRecord Bundle where
    parseRecord rec
        | V.length rec == 785 = do
            expectDigit <- parseField $ V.head rec
            datas       <- V.mapM parseField $ V.drop 1 rec
            return $ Bundle ( V.convert datas
                            , genExpect expectDigit)
        | otherwise           = fail "A record must be 785 items long"

mnistFromFile :: FilePath -> IO (Either String (Samples Float))
mnistFromFile file =
    either (return . Left)
           (return . Right . map unbundle . V.toList) =<<
                (decode NoHeader <$> LBS.readFile file)

unbundle :: Bundle -> Sample Float
unbundle (Bundle x) = x

genExpect :: Int -> VS.Vector Float
genExpect n = VS.generate 10 (\x -> if x == n then 1 else 0)
