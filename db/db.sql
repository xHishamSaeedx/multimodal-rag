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

-- Enable Row Level Security
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;

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

