import * as Minio from 'minio'

// Parse endpoint – supports "host:port" or just "host"
const minioEndpointRaw = process.env.MINIO_ENDPOINT ?? 'localhost:9000'
const [minioHost, minioPortStr] = minioEndpointRaw.includes(':')
  ? minioEndpointRaw.split(':')
  : [minioEndpointRaw, '9000']

export const minioClient = new Minio.Client({
  endPoint: minioHost,
  port: Number(minioPortStr),
  useSSL: process.env.MINIO_SECURE === 'true',
  accessKey: process.env.MINIO_ACCESS_KEY ?? 'minioadmin',
  secretKey: process.env.MINIO_SECRET_KEY ?? 'minioadmin123',
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