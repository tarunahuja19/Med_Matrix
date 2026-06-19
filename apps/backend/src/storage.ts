import * as Minio from 'minio'

export const minioClient = new Minio.Client({
  endPoint: 'localhost',
  port: 9000,
  useSSL: false,
  accessKey: 'minioadmin',
  secretKey: 'minioadmin123'
})

const BUCKETS = [
  'dicom-files',      // raw DICOM uploads
  'kspace-raw',       // raw K-Space .dat/.h5 files
  'reconstructed',    // IFFT reconstructed images
  'segmentation-masks', // NIfTI .nii.gz mask files
  'reports'           // generated PDF reports
]

export async function initBuckets() {
  for (const bucket of BUCKETS) {
    const exists = await minioClient.bucketExists(bucket)
    if (!exists) {
      await minioClient.makeBucket(bucket)
      console.log(`✓ Bucket created: ${bucket}`)
    } else {
      console.log(`✓ Bucket exists: ${bucket}`)
    }
  }
}