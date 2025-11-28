-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    source_path TEXT,
    filename TEXT,
    document_type TEXT,
    extracted_text TEXT,
    metadata JSONB DEFAULT '{}'::jsonb  -- author, tags, version, upload_date
);

-- Create chunks table
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    document_id UUID NOT NULL REFERENCES documents(id) ON UPDATE CASCADE ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_type TEXT DEFAULT 'text',
    table_data JSONB,  -- For table chunks (Phase 2)
    image_path TEXT,   -- Path to image file (Phase 2)
    image_caption TEXT, -- Optional caption for images (Phase 2)
    embedding_type TEXT DEFAULT 'text', -- text, table, image (Phase 2)
    metadata JSONB DEFAULT '{}'::jsonb  -- page_number, section, etc.
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS documents_source_path_idx ON documents(source_path);
CREATE INDEX IF NOT EXISTS documents_filename_idx ON documents(filename);
CREATE INDEX IF NOT EXISTS documents_document_type_idx ON documents(document_type);
CREATE INDEX IF NOT EXISTS documents_created_at_idx ON documents(created_at);
CREATE INDEX IF NOT EXISTS documents_updated_at_idx ON documents(updated_at);
CREATE INDEX IF NOT EXISTS documents_metadata_idx ON documents USING GIN(metadata);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks(document_id);
CREATE INDEX IF NOT EXISTS chunks_chunk_index_idx ON chunks(document_id, chunk_index);
CREATE INDEX IF NOT EXISTS chunks_chunk_type_idx ON chunks(chunk_type);
CREATE INDEX IF NOT EXISTS chunks_created_at_idx ON chunks(created_at);
CREATE INDEX IF NOT EXISTS chunks_metadata_idx ON chunks USING GIN(metadata);
CREATE INDEX IF NOT EXISTS chunks_embedding_type_idx ON chunks(embedding_type);
CREATE INDEX IF NOT EXISTS chunks_image_path_idx ON chunks(image_path) WHERE image_path IS NOT NULL;

-- Create images table (Phase 2: Multimodal Support)
CREATE TABLE IF NOT EXISTS images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    document_id UUID NOT NULL REFERENCES documents(id) ON UPDATE CASCADE ON DELETE CASCADE,
    chunk_id UUID REFERENCES chunks(id) ON UPDATE CASCADE ON DELETE SET NULL,
    image_path TEXT NOT NULL,  -- Supabase storage path: {document_id}/image_{timestamp}-{random}.{ext}
    image_type TEXT,  -- diagram, chart, photo, screenshot
    extracted_text TEXT,  -- OCR text if applicable
    caption TEXT,
    metadata JSONB DEFAULT '{}'::jsonb  -- dimensions, format, page_number
);

-- Create tables table (Phase 2: Multimodal Support)
CREATE TABLE IF NOT EXISTS tables (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    document_id UUID NOT NULL REFERENCES documents(id) ON UPDATE CASCADE ON DELETE CASCADE,
    chunk_id UUID REFERENCES chunks(id) ON UPDATE CASCADE ON DELETE SET NULL,
    table_data JSONB NOT NULL,  -- Structured table data
    table_markdown TEXT,  -- Markdown representation
    table_text TEXT,  -- Flattened text representation
    metadata JSONB DEFAULT '{}'::jsonb  -- row_count, col_count, headers
);

-- Create indexes for images table
CREATE INDEX IF NOT EXISTS images_document_id_idx ON images(document_id);
CREATE INDEX IF NOT EXISTS images_chunk_id_idx ON images(chunk_id);
CREATE INDEX IF NOT EXISTS images_image_path_idx ON images(image_path);
CREATE INDEX IF NOT EXISTS images_image_type_idx ON images(image_type);
CREATE INDEX IF NOT EXISTS images_created_at_idx ON images(created_at);
CREATE INDEX IF NOT EXISTS images_metadata_idx ON images USING GIN(metadata);

-- Create indexes for tables table
CREATE INDEX IF NOT EXISTS tables_document_id_idx ON tables(document_id);
CREATE INDEX IF NOT EXISTS tables_chunk_id_idx ON tables(chunk_id);
CREATE INDEX IF NOT EXISTS tables_created_at_idx ON tables(created_at);
CREATE INDEX IF NOT EXISTS tables_table_data_idx ON tables USING GIN(table_data);
CREATE INDEX IF NOT EXISTS tables_metadata_idx ON tables USING GIN(metadata);

-- Enable Row Level Security
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE images ENABLE ROW LEVEL SECURITY;
ALTER TABLE tables ENABLE ROW LEVEL SECURITY;

-- RLS Policies for documents table
-- Policy: Authenticated users can view all documents
CREATE POLICY "Documents - Authenticated users can view" ON documents
FOR SELECT TO authenticated
USING (true);

-- Policy: Authenticated users can insert documents
CREATE POLICY "Documents - Authenticated users can insert" ON documents
FOR INSERT TO authenticated
WITH CHECK (true);

-- Policy: Authenticated users can update documents
CREATE POLICY "Documents - Authenticated users can update" ON documents
FOR UPDATE TO authenticated
USING (true);

-- Policy: Authenticated users can delete documents
CREATE POLICY "Documents - Authenticated users can delete" ON documents
FOR DELETE TO authenticated
USING (true);

-- RLS Policies for chunks table
-- Policy: Authenticated users can view all chunks
CREATE POLICY "Chunks - Authenticated users can view" ON chunks
FOR SELECT TO authenticated
USING (true);

-- Policy: Authenticated users can insert chunks
CREATE POLICY "Chunks - Authenticated users can insert" ON chunks
FOR INSERT TO authenticated
WITH CHECK (true);

-- Policy: Authenticated users can update chunks
CREATE POLICY "Chunks - Authenticated users can update" ON chunks
FOR UPDATE TO authenticated
USING (true);

-- Policy: Authenticated users can delete chunks
CREATE POLICY "Chunks - Authenticated users can delete" ON chunks
FOR DELETE TO authenticated
USING (true);

-- RLS Policies for images table
-- Policy: Authenticated users can view all images
CREATE POLICY "Images - Authenticated users can view" ON images
FOR SELECT TO authenticated
USING (true);

-- Policy: Authenticated users can insert images
CREATE POLICY "Images - Authenticated users can insert" ON images
FOR INSERT TO authenticated
WITH CHECK (true);

-- Policy: Authenticated users can update images
CREATE POLICY "Images - Authenticated users can update" ON images
FOR UPDATE TO authenticated
USING (true);

-- Policy: Authenticated users can delete images
CREATE POLICY "Images - Authenticated users can delete" ON images
FOR DELETE TO authenticated
USING (true);

-- RLS Policies for tables table
-- Policy: Authenticated users can view all tables
CREATE POLICY "Tables - Authenticated users can view" ON tables
FOR SELECT TO authenticated
USING (true);

-- Policy: Authenticated users can insert tables
CREATE POLICY "Tables - Authenticated users can insert" ON tables
FOR INSERT TO authenticated
WITH CHECK (true);

-- Policy: Authenticated users can update tables
CREATE POLICY "Tables - Authenticated users can update" ON tables
FOR UPDATE TO authenticated
USING (true);

-- Policy: Authenticated users can delete tables
CREATE POLICY "Tables - Authenticated users can delete" ON tables
FOR DELETE TO authenticated
USING (true);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER 
SET search_path = public, pg_catalog
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$;

-- Create triggers for updated_at
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Note: Service role key bypasses RLS automatically
-- These policies are for future use when using authenticated users
-- For backend operations, use SUPABASE_SERVICE_ROLE_KEY which bypasses RLS

-- Allow service role to bypass RLS (if needed for direct SQL access)
-- Service role already bypasses RLS, but adding explicit policy for clarity
-- CREATE POLICY "Service role bypass" ON documents FOR ALL TO service_role USING (true);
-- CREATE POLICY "Service role bypass" ON chunks FOR ALL TO service_role USING (true);

-- Create storage bucket for document images
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'document-images',
  'document-images',
  false,
  10485760, -- 10MB limit for image files
  ARRAY['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp', 'image/tiff', 'image/svg+xml']
) ON CONFLICT (id) DO NOTHING;

-- Storage policies for document-images bucket
-- File path structure: {document_id}/image_{timestamp}-{random}.{ext}

-- Policy: Authenticated users can upload images to document folders
CREATE POLICY "Document Images - Upload" ON storage.objects
FOR INSERT TO authenticated
WITH CHECK (
  bucket_id = 'document-images' AND
  -- Allow authenticated users to upload images
  true
);

-- Policy: Authenticated users can view images from document folders
CREATE POLICY "Document Images - View" ON storage.objects
FOR SELECT TO authenticated
USING (
  bucket_id = 'document-images' AND
  -- Allow authenticated users to view images
  true
);

-- Policy: Authenticated users can update images in document folders
CREATE POLICY "Document Images - Update" ON storage.objects
FOR UPDATE TO authenticated
USING (
  bucket_id = 'document-images' AND
  -- Allow authenticated users to update images
  true
);

-- Policy: Authenticated users can delete images from document folders
CREATE POLICY "Document Images - Delete" ON storage.objects
FOR DELETE TO authenticated
USING (
  bucket_id = 'document-images' AND
  -- Allow authenticated users to delete images
  true
);

