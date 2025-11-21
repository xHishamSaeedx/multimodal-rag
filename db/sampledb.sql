-- Create profiles table (moved up to resolve dependency issues)
CREATE TABLE IF NOT EXISTS profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON UPDATE CASCADE ON DELETE CASCADE,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    UNIQUE(user_id)
);

-- Creae pricing_plans table
CREATE TABLE IF NOT EXISTS pricing_plans (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  name TEXT NOT NULL,  -- e.g., 'Starter', 'Growth', 'Enterprise', etc.
  min_project_size NUMERIC(15,2) NOT NULL,   -- in USD, e.g., 0 for Starter
  max_project_size NUMERIC(15,2) NOT NULL,   -- null for open-ended (Custom)
  base_fee NUMERIC(10,2) NOT NULL,
  billing_period TEXT NOT NULL DEFAULT 'monthly' CHECK (billing_period IN ('trial', 'monthly', 'annual')),
  description TEXT,
  features JSONB DEFAULT '[]'::jsonb,  -- Array of features included in this plan
  is_active BOOLEAN NOT NULL DEFAULT true,
  call_minutes_included INTEGER NOT NULL,
  phone_numbers_included INTEGER NOT NULL,
  tokens_included INTEGER NOT NULL
);

-- Create workspaces table
CREATE TABLE IF NOT EXISTS workspaces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    owner_id UUID NOT NULL REFERENCES auth.users(id) ON UPDATE CASCADE ON DELETE CASCADE,
    name TEXT NOT NULL,
    address TEXT,
    city TEXT,
    state TEXT,
    country TEXT NOT NULL,
    timezone TEXT DEFAULT 'Asia/Kolkata',
    is_active BOOLEAN DEFAULT true,
    type TEXT NOT NULL,
    currency CHAR(3) NOT NULL DEFAULT 'USD' CHECK (currency IN ('USD', 'EUR', 'GBP', 'INR', 'CAD', 'AUD', 'JPY', 'CHF', 'CNY', 'SEK', 'NOK', 'DKK', 'SGD', 'HKD', 'KRW', 'MXN', 'BRL', 'ZAR', 'RUB', 'TRY', 'PLN', 'CZK', 'HUF', 'RON', 'BGN', 'HRK', 'NZD', 'THB', 'MYR', 'PHP', 'IDR', 'VND', 'BDT', 'PKR', 'LKR', 'NPR', 'MMK', 'KHR', 'LAK', 'MNT', 'KZT', 'UZS', 'TJS', 'TMT', 'AZN', 'GEL', 'AMD', 'BYN', 'MDL', 'UAH', 'RSD', 'BAM', 'MKD', 'ALL', 'XCD', 'BBD', 'BMD', 'KYD', 'JMD', 'TTD', 'BZD', 'GTQ', 'HNL', 'NIO', 'CRC', 'PAB', 'PYG', 'UYU', 'CLP', 'ARS', 'BOB', 'PEN', 'COP', 'VEF', 'GYD', 'SRD', 'XOF', 'XAF', 'XPF', 'GHS', 'NGN', 'KES', 'UGX', 'TZS', 'ZMW', 'MWK', 'NAD', 'BWP', 'SZL', 'LSL', 'MUR', 'SCR', 'KMF', 'DJF', 'ETB', 'SOS', 'TND', 'DZD', 'MAD', 'LYD', 'EGP', 'SDG', 'SSP', 'ERN', 'RWF', 'BIF', 'CDF', 'GNF', 'MLF')),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Create projects table (moved up to resolve dependency issues)
CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    address TEXT,
    city TEXT,
    state TEXT,
    country TEXT,
    timezone TEXT DEFAULT 'UTC',
    start_date DATE,
    end_date DATE,
    status TEXT DEFAULT 'planned',                     -- e.g., "planned", "in_progress", "completed"
    project_type TEXT DEFAULT 'residential',               -- "residential", "commercial", "infrastructure"
    project_size NUMERIC,
    currency CHAR(3)
);

-- Create plivo_subaccounts table (moved up to resolve dependency issues)
CREATE TABLE IF NOT EXISTS plivo_subaccounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    plivo_subaccount_alias TEXT NOT NULL,
    plivo_subaccount_authid TEXT NOT NULL,
    plivo_subaccount_authtoken TEXT NOT NULL,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
    UNIQUE(workspace_id)
);

-- Create vendors table (moved up to resolve dependency issues)
CREATE TABLE IF NOT EXISTS vendors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
    project_id UUID REFERENCES projects(id) ON UPDATE CASCADE ON DELETE SET NULL,
    name TEXT NOT NULL,
    domain TEXT,
    vendor_type TEXT NOT NULL CHECK (vendor_type IN ('material', 'equipment', 'labor'))
);

-- Create agents table (moved up to resolve dependency issues)
CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE,
    name TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'voice',
    agent_type TEXT NOT NULL,
    status TEXT NOT NULL,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
    project_id UUID NOT NULL REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
    config JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Create contacts table (moved up to resolve dependency issues)
CREATE TABLE IF NOT EXISTS contacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
    project_id UUID REFERENCES projects(id) ON UPDATE CASCADE ON DELETE SET NULL,
    vendor_id UUID REFERENCES vendors(id) ON UPDATE CASCADE ON DELETE SET NULL,
    contact_type TEXT CHECK (contact_type IN ('team', 'vendor')),
    name TEXT NOT NULL,
    phone TEXT NOT NULL,
    email TEXT,
    role TEXT NOT NULL CHECK (role IN ('project_owner', 'financier', 'project_manager', 'site_supervisor', 'foreman_civil', 'foreman_electrical', 'foreman_plumbing', 'materials_manager', 'safety_officer', 'quality_engineer', 'planning_engineer', 'project_engineer', 'subcontractor_lead', 'supplier_cement', 'supplier_steel', 'supplier_bricks', 'supplier_aggregates', 'equipment_vendor', 'transport_vendor', 'other')),
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'archived')),
    is_india_number BOOLEAN DEFAULT false,
    language_preference TEXT DEFAULT 'en-US',
    UNIQUE(workspace_id, phone)
);

-- Create tasks table (moved up to resolve dependency issues)
CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
    agent_id UUID REFERENCES agents(id) ON UPDATE CASCADE ON DELETE SET NULL,
    project_id UUID NOT NULL REFERENCES projects(id) ON UPDATE CASCADE ON DELETE SET NULL,
    task_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    scheduled_for TIMESTAMP WITH TIME ZONE,
    cloud_task_name TEXT,
    contact_id UUID REFERENCES contacts(id) ON UPDATE CASCADE ON DELETE SET NULL,
    input_payload JSONB,
    output_payload JSONB,
    error_payload JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    assigned_to UUID REFERENCES profiles(id) ON UPDATE CASCADE ON DELETE SET NULL,
    is_agentic BOOLEAN DEFAULT TRUE,
    notes TEXT,
    reason TEXT,
    ref_task_id UUID, -- Removed FK to avoid circular dependency
    completed_at TIMESTAMP WITH TIME ZONE,
    project_action_id UUID -- Removed FK to avoid circular dependency
);

-- Pending model update
CREATE TABLE IF NOT EXISTS subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    workspace_id UUID REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE SET NULL,
    project_id UUID REFERENCES projects(id) ON UPDATE CASCADE ON DELETE SET NULL,
    pricing_plan_id UUID NOT NULL REFERENCES pricing_plans(id) ON UPDATE CASCADE ON DELETE RESTRICT,
    owner_id UUID NOT NULL REFERENCES auth.users(id) ON UPDATE CASCADE ON DELETE RESTRICT,
    subscription_start_date DATE NOT NULL DEFAULT CURRENT_DATE,
    rzp_subscription_id TEXT,
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'canceled', 'trial', 'past_due', 'unpaid')),
    UNIQUE(workspace_id, project_id, pricing_plan_id)
);

CREATE TABLE IF NOT EXISTS project_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
    project_id UUID REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
    call_minutes_used INTEGER,
    token_usage JSONB,
    tokens_used INTEGER DEFAULT 0
);

-- Create phone_numbers table
CREATE TABLE IF NOT EXISTS phone_numbers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
    plivo_subaccount_id UUID REFERENCES plivo_subaccounts(id) ON UPDATE CASCADE ON DELETE CASCADE,
    phone_number TEXT NOT NULL,
    country_code TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('local', 'tollfree', 'international')),
    capabilities JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'pending')),
    monthly_rental DECIMAL(10,2) NOT NULL,
    last_used_at TIMESTAMP WITH TIME ZONE,
    is_locked BOOLEAN DEFAULT false,
    is_assigned BOOLEAN DEFAULT false,
    UNIQUE(phone_number)
);

-- Create call_logs table
CREATE TABLE IF NOT EXISTS call_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    call_uuid TEXT NOT NULL,
    parent_call_uuid TEXT,
    agent_id UUID REFERENCES agents(id) ON UPDATE CASCADE ON DELETE SET NULL,
    project_id UUID REFERENCES projects(id) ON UPDATE CASCADE ON DELETE SET NULL,
    workspace_id UUID REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE SET NULL,
    -- Call timing metrics
    start_time TIMESTAMP WITH TIME ZONE,
    answer_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    -- Duration metrics
    call_duration INTEGER NOT NULL DEFAULT 0,
    bill_duration INTEGER,
    billed_duration INTEGER,
    -- Call direction and numbers
    call_direction TEXT CHECK (call_direction IN ('inbound', 'outbound')),
    from_number TEXT,
    to_number TEXT,
    -- Call status and events
    call_status TEXT,
    event_type TEXT,
    -- Hangup details
    hangup_cause TEXT,
    hangup_cause_code INTEGER,
    hangup_cause_name TEXT,
    hangup_source TEXT,
    -- Security and verification
    stir_verification TEXT CHECK (stir_verification IN ('Verified', 'Not Verified', 'Not Applicable')),
    source_ip TEXT,
    -- Billing information
    total_cost DECIMAL(10,4),
    bill_rate DECIMAL(10,4),
    -- Additional metadata
    resource_uri TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    is_test BOOLEAN DEFAULT false,
    task_id UUID REFERENCES tasks(id) ON UPDATE CASCADE ON DELETE SET NULL,
    plivo_recording_url TEXT,
    recording_path TEXT,
    transcript_path TEXT,
    original_transcript_path TEXT,
    user_id UUID REFERENCES profiles(user_id) ON UPDATE CASCADE ON DELETE SET NULL,
    is_processed BOOLEAN DEFAULT FALSE,
    UNIQUE(workspace_id, call_uuid)
);

-- Create workspace_members table for managing workspace access
CREATE TABLE IF NOT EXISTS workspace_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON UPDATE CASCADE ON DELETE CASCADE,
    role TEXT NOT NULL DEFAULT 'member' CHECK (role IN ('owner', 'admin', 'member')),
    UNIQUE(workspace_id, user_id)
);

-- Create project_member_access for managing project access
CREATE TABLE IF NOT EXISTS project_member_access (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
    project_id UUID NOT NULL REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON UPDATE CASCADE ON DELETE CASCADE,
    access_level TEXT NOT NULL DEFAULT 'view' CHECK (access_level IN ('none', 'view', 'edit')),
    granted_by UUID REFERENCES auth.users(id) ON UPDATE CASCADE ON DELETE SET NULL,
    UNIQUE(workspace_id, project_id, user_id)
);

-- Create plivo_endpoints table
CREATE TABLE IF NOT EXISTS plivo_endpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
    agent_id UUID REFERENCES agents(id) ON UPDATE CASCADE ON DELETE CASCADE,
    member_id UUID REFERENCES profiles(id) ON UPDATE CASCADE ON DELETE CASCADE,
    username TEXT NOT NULL,
    password TEXT NOT NULL,
    endpoint_id TEXT NOT NULL,
    alias TEXT NOT NULL,
    UNIQUE(agent_id),
    UNIQUE(member_id)
);

-- Create phone_pools table
CREATE TABLE IF NOT EXISTS phone_pools (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
    project_id UUID REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Create phone_pool_memberships table
CREATE TABLE IF NOT EXISTS phone_pool_memberships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    phone_pool_id UUID NOT NULL REFERENCES phone_pools(id) ON UPDATE CASCADE ON DELETE CASCADE,
    phone_number_id UUID NOT NULL REFERENCES phone_numbers(id) ON UPDATE CASCADE ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    UNIQUE(phone_pool_id, phone_number_id)
);

-- Create agent_phone_mappings table
CREATE TABLE IF NOT EXISTS agent_phone_mappings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
    agent_id UUID NOT NULL REFERENCES agents(id) ON UPDATE CASCADE ON DELETE CASCADE,
    phone_number_id UUID NOT NULL REFERENCES phone_numbers(id) ON UPDATE CASCADE ON DELETE CASCADE,
    UNIQUE(agent_id, phone_number_id)
);

CREATE TABLE IF NOT EXISTS kb_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
    project_id UUID NOT NULL REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    mime_type TEXT,
    tags TEXT[],
    filename TEXT,
    description TEXT
);

-- Create call_analysis_results table
CREATE TABLE IF NOT EXISTS call_analysis_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
    project_id UUID NOT NULL REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
    call_uuid TEXT NOT NULL,
    -- Analysis results
    work_progress JSONB,
    delays JSONB,
    rfis JSONB,
    safety_incidents JSONB,
    action_items JSONB,
    issues JSONB,
    materials JSONB,
    equipment JSONB,
    labor JSONB,
    -- Metadata
    analysis_status TEXT DEFAULT 'completed' CHECK (analysis_status IN ('pending', 'in_progress', 'completed', 'failed')),
    error_message TEXT,
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    -- Indexes for performance
    UNIQUE(workspace_id, project_id, call_uuid)
);

CREATE TABLE IF NOT EXISTS project_schedules (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
  project_id UUID NOT NULL REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
  location TEXT,
  activity_type TEXT NOT NULL,
  activity_category TEXT CHECK (activity_category IN ('Foundation', 'Structural', 'MEP', 'Electrical', 'Plumbing', 'HVAC', 'Finishing', 'Interior', 'Exterior', 'Roofing', 'Other')),
  activity_description TEXT,
  schedule_code TEXT UNIQUE, -- optional
  planned_start DATE NOT NULL,
  planned_end DATE NOT NULL,
  baseline_start DATE,
  baseline_end DATE,
  actual_start DATE,
  actual_end DATE,
  status TEXT NOT NULL DEFAULT 'planned' CHECK (status IN ('planned', 'in_progress', 'completed', 'delayed', 'cancelled')),
  priority TEXT DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high', 'critical')),
  assigned_to UUID REFERENCES project_contacts(id) ON UPDATE CASCADE ON DELETE SET NULL,
  vendor_id UUID REFERENCES vendors(id) ON UPDATE CASCADE ON DELETE SET NULL,
  progress_percentage INTEGER DEFAULT 0 CHECK (progress_percentage >= 0 AND progress_percentage <= 100),
  notes TEXT,
  milestone BOOLEAN DEFAULT FALSE,
  schedule_source TEXT DEFAULT 'manual',
  created_by UUID REFERENCES profiles(id) ON UPDATE CASCADE ON DELETE SET NULL,
  crew_size INTEGER,
  work_hours NUMERIC(10,2) DEFAULT 0,
  duration_days INTEGER GENERATED ALWAYS AS (planned_end - planned_start) STORED,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS project_schedule_dependencies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),

    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
    project_id UUID NOT NULL REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,

    schedule_id UUID NOT NULL REFERENCES project_schedules(id) ON UPDATE CASCADE ON DELETE CASCADE,
    depends_on_schedule_id UUID NOT NULL REFERENCES project_schedules(id) ON UPDATE CASCADE ON DELETE CASCADE,

    type TEXT DEFAULT 'FS' CHECK (type IN ('FS', 'SS', 'FF', 'SF')), -- Finish-Start, Start-Start etc
    lag_days INTEGER DEFAULT 0, -- lag or lead time between dependencies
    notes TEXT
);

CREATE TABLE IF NOT EXISTS project_budgets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
  project_id UUID NOT NULL REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
  category TEXT NOT NULL CHECK (category IN (
    'site_preparation', 'excavation', 'foundation', 'structural', 'mep', 'electrical', 
    'plumbing', 'hvac', 'fire_safety', 'finishing', 'interior', 'exterior', 'roofing', 
    'landscaping', 'paving', 'utilities', 'equipment', 'labor', 'subcontractor', 
    'permits', 'inspections', 'insurance', 'legal', 'professional_services', 
    'marketing', 'overhead', 'contingency', 'escalation', 'misc'
  )),
  sub_category TEXT, -- Free text field for PM flexibility
  budget_type TEXT DEFAULT 'direct' CHECK (budget_type IN ('direct', 'indirect', 'contingency', 'escalation')),
  phase TEXT, -- Project phase (e.g., 'pre_construction', 'construction', 'closeout')
  planned_amount NUMERIC(15,2) DEFAULT 0,
  actual_amount NUMERIC(15,2) DEFAULT 0,
  committed_amount NUMERIC(15,2) DEFAULT 0, -- Amount committed but not yet spent
  remaining_amount NUMERIC(15,2) GENERATED ALWAYS AS (planned_amount - actual_amount - committed_amount) STORED,
  variance NUMERIC(15,2) GENERATED ALWAYS AS (planned_amount - actual_amount) STORED,
  variance_percentage NUMERIC(5,2) GENERATED ALWAYS AS (
    CASE 
      WHEN planned_amount > 0 THEN ((planned_amount - actual_amount) / planned_amount * 100)
      ELSE 0 
    END
  ) STORED,
  status TEXT DEFAULT 'active' CHECK (status IN ('active', 'on_hold', 'completed', 'cancelled')),
  priority TEXT DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high', 'critical')),
  responsible_party TEXT, -- Who is responsible for this budget item
  contact_id UUID REFERENCES contacts(id) ON UPDATE CASCADE ON DELETE SET NULL, -- Associated contact
  vendor_id UUID REFERENCES vendors(id) ON UPDATE CASCADE ON DELETE SET NULL,
  start_date DATE,
  end_date DATE,
  notes TEXT,
  created_by UUID REFERENCES auth.users(id) ON UPDATE CASCADE ON DELETE SET NULL,
  last_modified_by UUID REFERENCES auth.users(id) ON UPDATE CASCADE ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS vendors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
    project_id UUID REFERENCES projects(id) ON UPDATE CASCADE ON DELETE SET NULL,
    name TEXT NOT NULL,
    domain TEXT,
    vendor_type TEXT NOT NULL CHECK (vendor_type IN ('material', 'equipment', 'labor'))
);

CREATE TABLE IF NOT EXISTS vendor_rates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    vendor_id UUID NOT NULL REFERENCES vendors(id) ON UPDATE CASCADE ON DELETE CASCADE,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
    project_id UUID REFERENCES projects(id) ON UPDATE CASCADE ON DELETE SET NULL,
    rate_type TEXT NOT NULL CHECK (rate_type IN ('hourly', 'daily', 'per_unit', 'fixed', 'lump_sum')),
    unit TEXT,
    rate NUMERIC(10,2) NOT NULL,
    effective_from DATE NOT NULL,
    effective_to DATE,
    is_active BOOLEAN DEFAULT true
);

CREATE TABLE IF NOT EXISTS material_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
    project_id UUID NOT NULL REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
    schedule_id UUID REFERENCES project_schedules(id) ON UPDATE CASCADE ON DELETE SET NULL,
    material_name TEXT NOT NULL,
    quantity_delivered NUMERIC(10,2) DEFAULT 0,
    unit TEXT,
    delivery_date DATE,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'ordered', 'delivered', 'cancelled', 'consumed')),
    vendor_id UUID REFERENCES vendors(id) ON UPDATE CASCADE ON DELETE SET NULL,
    vendor_name TEXT,
    notes TEXT,
    is_ai BOOLEAN DEFAULT TRUE,
    is_processed BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS equipment_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
    project_id UUID NOT NULL REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
    schedule_id UUID REFERENCES project_schedules(id) ON UPDATE CASCADE ON DELETE SET NULL,
    equipment_type TEXT NOT NULL,
    vendor_id UUID REFERENCES vendors(id) ON UPDATE CASCADE ON DELETE SET NULL,
    vendor_name TEXT,
    usage_date DATE NOT NULL,
    usage_hours NUMERIC(5,2) DEFAULT 0,
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'maintenance', 'broken', 'returned')),
    location TEXT,
    assigned_to TEXT,
    remarks TEXT,
    is_ai BOOLEAN DEFAULT TRUE,
    is_processed BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS labor_tracking (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
  project_id UUID NOT NULL REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
  schedule_id UUID,
  worker_type TEXT NOT NULL,
  worker_id UUID REFERENCES contacts(id) ON UPDATE CASCADE ON DELETE SET NULL,
  worker_name TEXT,
  headcount INTEGER DEFAULT 1,
  work_date DATE NOT NULL,
  hours_worked NUMERIC(5,2) NOT NULL,
  activity TEXT,        -- "Formwork Installation", "Slab Work", "Bathroom Fittings"
  remarks TEXT,
  is_ai BOOLEAN DEFAULT TRUE,
  is_processed BOOLEAN DEFAULT FALSE
);

CREATE TYPE alert_type_enum AS ENUM (
  'cost_variance',
  'change_order_logged',
  'material_price_spike',
  'equipment_rate_increase',
  'labor_rate_deviation',
  'unplanned_site_condition',
  'design_change_cost',
  'schedule_delay',
  'permit_fee_variation',
  'safety_incident'
);

CREATE TABLE IF NOT EXISTS project_alerts (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  workspace_id     UUID NOT NULL REFERENCES workspaces(id)    ON UPDATE CASCADE ON DELETE CASCADE,
  project_id       UUID NOT NULL REFERENCES projects(id)      ON UPDATE CASCADE ON DELETE CASCADE,
  alert_type       alert_type_enum NOT NULL,                  -- budget, safety, delay, etc.
  context_table    TEXT,                                       -- e.g. 'project_budgets'
  context_id       UUID,                                       -- optional FK to specific row
  description      TEXT NOT NULL,
  suggested_action TEXT,
  severity         TEXT    DEFAULT 'medium' CHECK (severity IN ('low','medium','high','critical')),
  status           TEXT    DEFAULT 'active' CHECK (status IN ('active','acknowledged','resolved')),
  generated_by     TEXT    CHECK (generated_by IN ('AI','manual')) DEFAULT 'AI',
  metadata         JSONB   DEFAULT '{}'::jsonb,
  created_at       TIMESTAMPTZ DEFAULT now(),
  updated_at       TIMESTAMPTZ DEFAULT now(),
  resolved_at      TIMESTAMPTZ,
  resolved_by      UUID    REFERENCES profiles(id) ON UPDATE CASCADE ON DELETE SET NULL,
  show_to_user_ids UUID[]
);

-- 1) Enum for recommendation domains
CREATE TYPE recommendation_type_enum AS ENUM (
  'action_item',
  'schedule_adjustment',     -- e.g. shift dates or re-sequence tasks
  'delay_mitigation',        -- e.g. add crew, change sequence
  'budget_alert',            -- e.g. budget alerts
  'risk_alert',              -- e.g. permit, design, site‐condition risks
  'safety_action',           -- e.g. schedule safety meeting, issue PPE
  'cost_saving',             -- e.g. suggest cheaper material/vendor
  'resource_allocation',     -- e.g. rebalance labor or equipment
  'custom'                   -- catch-all for future types
);

-- 2) Enum for recommendation status
CREATE TYPE recommendation_status_enum AS ENUM (
  'pending',    -- newly generated, awaiting review
  'accepted',   -- user has approved and (optionally) applied it
  'rejected'    -- user declined or marked as not applicable
);

-- 3) Core recommendations table
CREATE TABLE IF NOT EXISTS ai_recommendations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
  project_id UUID NOT NULL REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
  -- what kind of recommendation
  recommendation_type recommendation_type_enum NOT NULL,
  -- free-text summary / title
  summary TEXT NOT NULL,
  -- detailed payload (dates, amounts, steps, etc.)
  details JSONB NOT NULL,
  -- optional linkage to a specific record (e.g. a schedule or alert)
  context_table TEXT,    -- e.g. 'project_schedules', 'budget_alerts'
  context_id UUID,       -- the specific row that this pertains to
  -- user feedback & lifecycle
  status recommendation_status_enum NOT NULL DEFAULT 'pending',
  reviewed_by UUID REFERENCES profiles(id) ON UPDATE CASCADE ON DELETE SET NULL,      -- FK → profiles(id), NULL until someone reviews 
  reviewed_at TIMESTAMP WITH TIME ZONE, 
  user_comments TEXT,     -- optional explanatory feedback
  generated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS project_delays (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
  project_id UUID NOT NULL REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
  delay_date DATE NOT NULL,
  delay_type TEXT CHECK (delay_type IN ('weather', 'material', 'equipment', 'labor', 'permit', 'design', 'site_condition', 'vendor', 'quality', 'safety', 'subcontractor', 'utility', 'inspection', 'other')),
  delay_category TEXT CHECK (delay_category IN ('excusable', 'non_excusable', 'compensable', 'non_compensable')),
  delayed_by INTEGER, -- Duration in hours
  reason TEXT,
  location TEXT,
  affected_activities TEXT[],
  impact TEXT,
  cost_impact NUMERIC(15,2),
  cost_currency TEXT DEFAULT 'USD' CHECK (cost_currency IN ('USD', 'EUR', 'GBP', 'INR', 'CAD', 'AUD', 'JPY', 'CHF', 'CNY', 'SEK', 'NOK', 'DKK', 'SGD', 'HKD', 'KRW', 'MXN', 'BRL', 'ZAR', 'RUB', 'TRY', 'PLN', 'CZK', 'HUF', 'RON', 'BGN', 'HRK', 'NZD', 'THB', 'MYR', 'PHP', 'IDR', 'VND', 'BDT', 'PKR', 'LKR', 'NPR', 'MMK', 'KHR', 'LAK', 'MNT', 'KZT', 'UZS', 'TJS', 'TMT', 'AZN', 'GEL', 'AMD', 'BYN', 'MDL', 'UAH', 'RSD', 'BAM', 'MKD', 'ALL', 'XCD', 'BBD', 'BMD', 'KYD', 'JMD', 'TTD', 'BZD', 'GTQ', 'HNL', 'NIO', 'CRC', 'PAB', 'PYG', 'UYU', 'CLP', 'ARS', 'BOB', 'PEN', 'COP', 'VEF', 'GYD', 'SRD', 'XOF', 'XAF', 'XPF', 'GHS', 'NGN', 'KES', 'UGX', 'TZS', 'ZMW', 'MWK', 'NAD', 'BWP', 'SZL', 'LSL', 'MUR', 'SCR', 'KMF', 'DJF', 'ETB', 'SOS', 'TND', 'DZD', 'MAD', 'LYD', 'EGP', 'SDG', 'SSP', 'ERN', 'RWF', 'BIF', 'CDF', 'GNF', 'MLF')),
  mitigation_plan TEXT,
  responsible_party_type TEXT CHECK (responsible_party_type IN ('internal_team', 'vendor_company', 'vendor_contact', 'subcontractor', 'client', 'regulatory', 'weather', 'other')),
  responsible_party_name TEXT,
  responsible_party_id UUID, -- Can reference profiles(id), contacts(id), or vendors(id) based on type
  escalation_needed BOOLEAN DEFAULT false,
  priority TEXT DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high', 'critical')),
  summary TEXT,
  status TEXT DEFAULT 'active' CHECK (status IN ('active', 'resolved', 'mitigated')),
  is_ai BOOLEAN DEFAULT TRUE,
  is_processed BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS project_rfis (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
  project_id UUID NOT NULL REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
  rfi_date DATE NOT NULL,
  rfi_type TEXT CHECK (rfi_type IN ('design', 'material', 'equipment', 'coordination', 'site_condition', 'quality', 'safety', 'code_compliance', 'installation_method', 'scope_clarification', 'technical_specification', 'other')),
  subject TEXT NOT NULL,
  location TEXT,
  requested_by UUID REFERENCES contacts(id) ON UPDATE CASCADE ON DELETE SET NULL,
  priority TEXT DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high', 'critical')),
  description TEXT NOT NULL,
  impact TEXT,
  summary TEXT,
  status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'submitted', 'answered', 'closed')),
  assigned_to UUID REFERENCES project_contacts(id) ON UPDATE CASCADE ON DELETE SET NULL,
  due_date DATE,
  response TEXT,
  is_ai BOOLEAN DEFAULT TRUE,
  is_processed BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS project_safety_incidents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
  project_id UUID NOT NULL REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
  incident_date DATE NOT NULL,
  incident_type TEXT CHECK (incident_type IN ('injury', 'near_miss', 'property_damage', 'environmental', 'security', 'equipment_failure', 'fire', 'chemical_spill', 'electrical', 'fall_protection', 'other')),
  severity TEXT DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
  location TEXT,
  description TEXT NOT NULL,
  immediate_action TEXT,
  root_cause TEXT,
  preventive_measures TEXT,
  reported_by UUID REFERENCES contacts(id) ON UPDATE CASCADE ON DELETE SET NULL,
  summary TEXT,
  status TEXT DEFAULT 'open' CHECK (status IN ('open', 'investigating', 'resolved', 'closed')),
  assigned_to UUID REFERENCES project_contacts(id) ON UPDATE CASCADE ON DELETE SET NULL,
  investigation_complete BOOLEAN DEFAULT false,
  cost_impact NUMERIC(15,2),
  is_ai BOOLEAN DEFAULT TRUE,
  is_processed BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS project_actions (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  workspace_id      UUID NOT NULL REFERENCES workspaces(id)    ON UPDATE CASCADE ON DELETE CASCADE,
  project_id        UUID NOT NULL REFERENCES projects(id)      ON UPDATE CASCADE ON DELETE CASCADE,
  call_uuid         TEXT,
  action_item_date DATE,
  ai_recommendation_id UUID REFERENCES ai_recommendations(id) ON UPDATE CASCADE ON DELETE SET NULL,
  task_id           UUID REFERENCES tasks(id) ON UPDATE CASCADE ON DELETE SET NULL,       
  assigned_to       UUID   REFERENCES project_contacts(id) ON UPDATE CASCADE ON DELETE SET NULL,
  context           TEXT,                   -- e.g. "Pour footing at Tower A" or free-form
  priority          TEXT   NOT NULL DEFAULT 'medium' CHECK (priority IN ('low','medium','high','critical')),
  status            TEXT   NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','in_progress','completed','cancelled')),
  deadline          DATE,
  created_by        UUID   REFERENCES contacts(id) ON UPDATE CASCADE ON DELETE SET NULL,
  created_at        TIMESTAMPTZ DEFAULT now(),
  updated_at        TIMESTAMPTZ DEFAULT now(),
  is_ai BOOLEAN DEFAULT TRUE,
  is_processed BOOLEAN DEFAULT FALSE
);


CREATE TABLE IF NOT EXISTS project_issues (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
  project_id UUID NOT NULL REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
  issue_date DATE,
  issue TEXT NOT NULL,
  severity TEXT DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
  location TEXT,
  immediate_action_needed BOOLEAN DEFAULT false,
  status TEXT DEFAULT 'open' CHECK (status IN ('open', 'in_progress', 'resolved', 'closed')),
  assigned_to UUID REFERENCES project_contacts(id) ON UPDATE CASCADE ON DELETE SET NULL,
  description TEXT,
  resolution TEXT,
  created_by UUID REFERENCES contacts(id) ON UPDATE CASCADE ON DELETE SET NULL,
  is_ai BOOLEAN DEFAULT TRUE,
  is_processed BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS project_contacts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
  project_id UUID NOT NULL REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
  contact_id UUID NOT NULL REFERENCES contacts(id) ON UPDATE CASCADE ON DELETE CASCADE,
  project_role TEXT CHECK (project_role IN ('project_owner', 'financier', 'project_manager', 'site_supervisor', 'foreman_civil', 'foreman_electrical', 'foreman_plumbing', 'materials_manager', 'safety_officer', 'quality_engineer', 'planning_engineer', 'project_engineer', 'subcontractor_lead', 'supplier_cement', 'supplier_steel', 'supplier_bricks', 'supplier_aggregates', 'equipment_vendor', 'transport_vendor', 'architect', 'structural_engineer', 'mechanical_engineer', 'electrical_engineer', 'plumbing_engineer', 'hvac_engineer', 'surveyor', 'quantity_surveyor', 'contractor', 'building_inspector', 'environmental_officer', 'legal_advisor', 'insurance_agent', 'other')),
  UNIQUE(project_id, contact_id)
);

CREATE TABLE IF NOT EXISTS dpr_call_settings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
  project_id UUID NOT NULL REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
  project_contact_id UUID NOT NULL REFERENCES project_contacts(id) ON UPDATE CASCADE ON DELETE CASCADE,
  daily_call_time TIME NOT NULL,
  from_date DATE NOT NULL,
  to_date DATE NOT NULL,
  enabled BOOLEAN NOT NULL DEFAULT true,
  UNIQUE(workspace_id, project_id, project_contact_id)
);

CREATE TABLE IF NOT EXISTS project_work_progress (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
  project_id UUID NOT NULL REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
  call_uuid TEXT,
  activity_date DATE NOT NULL,
  activity_type TEXT NOT NULL,
  activity_category TEXT CHECK (activity_category IN ('Foundation', 'Structural', 'MEP', 'Electrical', 'Plumbing', 'HVAC', 'Finishing', 'Interior', 'Exterior', 'Roofing', 'Other')),
  location TEXT,
  progress_percentage INTEGER CHECK (progress_percentage >= 0 AND progress_percentage <= 100),
  status TEXT,
  crew_size INTEGER,
  materials_used TEXT[],
  equipment_used TEXT[],
  notes TEXT,
  next_steps TEXT,
  summary TEXT,
  is_processed BOOLEAN DEFAULT FALSE,
  is_ai BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS work_progress (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
  project_id UUID NOT NULL REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
  activity_date DATE NOT NULL,
  activity_category TEXT CHECK (activity_category IN ('Foundation', 'Structural', 'MEP', 'Mechanical', 'Electrical', 'Plumbing', 'HVAC', 'Finishing', 'Interior', 'Exterior', 'Roofing', 'Other')),
  activity_type TEXT NOT NULL,
  summary TEXT,
  location TEXT,
  status TEXT,
  reported_by UUID REFERENCES project_contacts(id) ON UPDATE CASCADE ON DELETE SET NULL,
  materials_consumed JSONB DEFAULT '[]'::jsonb,
  equipment_used JSONB DEFAULT '[]'::jsonb,
  labor_employed JSONB DEFAULT '[]'::jsonb
);

CREATE TABLE IF NOT EXISTS issues (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
  project_id UUID NOT NULL REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
  reported_by UUID REFERENCES auth.users(id) ON UPDATE CASCADE ON DELETE SET NULL,
  title TEXT NOT NULL,
  description TEXT,
  schedule_id UUID,
  work_front TEXT,
  status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
  need_by TIMESTAMP WITH TIME ZONE,
  due_time TIMESTAMP WITH TIME ZONE,
  priority TEXT CHECK (priority IN ('low', 'medium', 'high')),
  is_blocking BOOLEAN DEFAULT false,
  attachment_paths TEXT[] DEFAULT ARRAY[]::TEXT[],
  photo_paths TEXT[] DEFAULT ARRAY[]::TEXT[],
  tag TEXT NOT NULL CHECK (tag IN ('rfi', 'hindrance', 'material_need', 'equipment_need', 'equipment_breakdown')),
  tag_data JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Create site_media table
CREATE TABLE IF NOT EXISTS site_media (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
  project_id UUID NOT NULL REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
  file_path TEXT NOT NULL,
  filename TEXT NOT NULL,
  mime_type TEXT,
  file_size INTEGER,
  description TEXT,
  location TEXT,
  capture_date DATE,
  media_category TEXT CHECK (media_category IN ('progress', 'safety', 'quality', 'defect', 'completion', 'milestone', 'inspection', 'material', 'equipment', 'before_after', 'rework', 'site_condition', 'access', 'storage', 'waste', 'as_built', 'drawing', 'permit', 'inspection_report', 'ppe', 'barricade', 'emergency', 'other')),
  tags TEXT[],
  uploaded_by UUID REFERENCES auth.users(id) ON UPDATE CASCADE ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS project_cashflows (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
  project_id UUID NOT NULL REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
  budget_id UUID REFERENCES project_budgets(id) ON UPDATE CASCADE ON DELETE SET NULL,
  contact_id UUID REFERENCES contacts(id) ON UPDATE CASCADE ON DELETE SET NULL,
  user_id UUID REFERENCES auth.users(id) ON UPDATE CASCADE ON DELETE SET NULL,
  flow_type TEXT NOT NULL CHECK (flow_type IN ('cash_in', 'cash_out')),
  amount NUMERIC(15,2) NOT NULL,
  category TEXT NOT NULL,
  sub_category TEXT,
  payment_mode TEXT CHECK (payment_mode IN ('cash', 'bank_transfer', 'upi', 'cheque', 'card')),
  reference_no TEXT,
  file_path TEXT,
  notes TEXT
);

-- Enable Row Level Security for project_cashflows
ALTER TABLE project_cashflows ENABLE ROW LEVEL SECURITY;

-- RLS Policies for project_cashflows table
-- Policy: Workspace owners/admins have full access to all cashflows in their workspaces
CREATE POLICY "Project Cashflows - Owner and admin access" ON project_cashflows
FOR ALL TO authenticated
USING (
  workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
);

-- Policy: Members can view cashflows based on project access level
CREATE POLICY "Project Cashflows - Member view access" ON project_cashflows
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_cashflows.workspace_id
    AND pma.project_id = project_cashflows.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level IN ('view', 'edit')
  )
);

-- Policy: Members can insert cashflows if they have edit access to the project
CREATE POLICY "Project Cashflows - Member insert access" ON project_cashflows
FOR INSERT TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_cashflows.workspace_id
    AND pma.project_id = project_cashflows.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can update cashflows if they have edit access to the project
CREATE POLICY "Project Cashflows - Member update access" ON project_cashflows
FOR UPDATE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_cashflows.workspace_id
    AND pma.project_id = project_cashflows.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can delete cashflows if they have edit access to the project
CREATE POLICY "Project Cashflows - Member delete access" ON project_cashflows
FOR DELETE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_cashflows.workspace_id
    AND pma.project_id = project_cashflows.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Create sla_settings table
CREATE TABLE IF NOT EXISTS sla_settings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
  project_id UUID REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
  entity_type TEXT NOT NULL CHECK (entity_type IN ('task', 'issue', 'handoff', 'approval')),
  entity_subtype TEXT,
  trigger_name TEXT NOT NULL,
  trigger_condition JSONB NOT NULL DEFAULT '{}'::jsonb,
  sla_minutes INTEGER,
  offset_from_due_minutes INTEGER,
  ai_action_enabled BOOLEAN DEFAULT false
);

-- Create subtasks table
CREATE TABLE IF NOT EXISTS subtasks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  workspace_id UUID NOT NULL REFERENCES workspaces(id) ON UPDATE CASCADE ON DELETE CASCADE,
  project_id UUID NOT NULL REFERENCES projects(id) ON UPDATE CASCADE ON DELETE CASCADE,
  task_id UUID REFERENCES tasks(id) ON UPDATE CASCADE ON DELETE CASCADE,
  issue_id UUID REFERENCES issues(id) ON UPDATE CASCADE ON DELETE CASCADE,
  sequence_number INTEGER NOT NULL,
  title TEXT NOT NULL,
  description TEXT,
  due_time TIMESTAMP WITH TIME ZONE,
  assigned_to UUID REFERENCES profiles(id) ON UPDATE CASCADE ON DELETE SET NULL,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'cancelled')),
  CHECK ((task_id IS NOT NULL AND issue_id IS NULL) OR (task_id IS NULL AND issue_id IS NOT NULL))
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS agents_workspace_id_idx ON agents(workspace_id);
CREATE INDEX IF NOT EXISTS agents_project_id_idx ON agents(project_id);
CREATE INDEX IF NOT EXISTS agents_status_category_idx ON agents(status, category);
CREATE INDEX IF NOT EXISTS agents_status_idx ON agents(status);
CREATE INDEX IF NOT EXISTS agents_category_idx ON agents(category);
CREATE INDEX IF NOT EXISTS agents_agent_type_idx ON agents(agent_type);
CREATE INDEX IF NOT EXISTS plivo_endpoints_agent_id_idx ON plivo_endpoints(agent_id);
CREATE INDEX IF NOT EXISTS phone_numbers_workspace_id_idx ON phone_numbers(workspace_id);
CREATE INDEX IF NOT EXISTS phone_numbers_status_idx ON phone_numbers(status);
CREATE INDEX IF NOT EXISTS agent_phone_mappings_agent_id_idx ON agent_phone_mappings(agent_id);
CREATE INDEX IF NOT EXISTS agent_phone_mappings_phone_number_id_idx ON agent_phone_mappings(phone_number_id);
CREATE INDEX IF NOT EXISTS call_logs_agent_id_idx ON call_logs(agent_id);
CREATE INDEX IF NOT EXISTS call_logs_project_id_idx ON call_logs(project_id);
CREATE INDEX IF NOT EXISTS call_logs_workspace_id_idx ON call_logs(workspace_id);
CREATE INDEX IF NOT EXISTS call_logs_call_uuid_idx ON call_logs(call_uuid);
CREATE INDEX IF NOT EXISTS call_logs_event_type_idx ON call_logs(event_type);

CREATE INDEX IF NOT EXISTS tasks_workspace_id_idx ON tasks(workspace_id);
CREATE INDEX IF NOT EXISTS tasks_status_idx ON tasks(status);
CREATE INDEX IF NOT EXISTS tasks_type_idx ON tasks(task_type);
CREATE INDEX IF NOT EXISTS task_agent_idx ON tasks(agent_id);
CREATE INDEX IF NOT EXISTS task_project_idx ON tasks(project_id);
CREATE INDEX IF NOT EXISTS tasks_scheduled_for_idx ON tasks(scheduled_for);
CREATE INDEX IF NOT EXISTS tasks_assigned_to_idx ON tasks(assigned_to);
CREATE INDEX IF NOT EXISTS tasks_is_agentic_idx ON tasks(is_agentic);
CREATE INDEX IF NOT EXISTS tasks_completed_at_idx ON tasks(completed_at);
CREATE INDEX IF NOT EXISTS tasks_input_payload_idx ON tasks USING GIN(input_payload);
CREATE INDEX IF NOT EXISTS tasks_output_payload_idx ON tasks USING GIN(output_payload);
CREATE INDEX IF NOT EXISTS tasks_error_payload_idx ON tasks USING GIN(error_payload);
CREATE INDEX IF NOT EXISTS call_logs_call_direction_idx ON call_logs(call_direction);
CREATE INDEX IF NOT EXISTS call_logs_end_time_idx ON call_logs(end_time);
CREATE INDEX IF NOT EXISTS call_logs_hangup_source_idx ON call_logs(hangup_source);
CREATE INDEX IF NOT EXISTS call_logs_stir_verification_idx ON call_logs(stir_verification);
CREATE INDEX IF NOT EXISTS contacts_workspace_id_idx ON contacts(workspace_id);
CREATE INDEX IF NOT EXISTS contacts_status_idx ON contacts(status);
CREATE INDEX IF NOT EXISTS contacts_project_id_idx ON contacts(project_id);
CREATE INDEX IF NOT EXISTS contacts_contact_type_idx ON contacts(contact_type);
CREATE INDEX IF NOT EXISTS projects_workspace_id_idx ON projects(workspace_id);
CREATE INDEX IF NOT EXISTS projects_active_idx ON projects(is_active);
CREATE INDEX IF NOT EXISTS projects_project_type_idx ON projects(project_type);
CREATE INDEX IF NOT EXISTS projects_project_size_idx ON projects(project_size);
CREATE INDEX IF NOT EXISTS projects_status_idx ON projects(status);
CREATE INDEX IF NOT EXISTS projects_start_date_idx ON projects(start_date);
CREATE INDEX IF NOT EXISTS projects_end_date_idx ON projects(end_date);
CREATE INDEX IF NOT EXISTS projects_dates_range_idx ON projects(start_date, end_date);
CREATE INDEX IF NOT EXISTS kb_files_workspace_id_idx ON kb_files(workspace_id);
CREATE INDEX IF NOT EXISTS kb_files_project_id_idx ON kb_files(project_id);
CREATE INDEX IF NOT EXISTS kb_files_tags_idx ON kb_files USING GIN(tags);
CREATE INDEX IF NOT EXISTS call_analysis_results_workspace_id_idx ON call_analysis_results(workspace_id);
CREATE INDEX IF NOT EXISTS call_analysis_results_project_id_idx ON call_analysis_results(project_id);
CREATE INDEX IF NOT EXISTS call_analysis_results_call_uuid_idx ON call_analysis_results(call_uuid);
CREATE INDEX IF NOT EXISTS call_analysis_results_status_idx ON call_analysis_results(analysis_status);
CREATE INDEX IF NOT EXISTS call_analysis_results_processed_at_idx ON call_analysis_results(processed_at);
CREATE INDEX IF NOT EXISTS project_schedules_workspace_id_idx ON project_schedules(workspace_id);
CREATE INDEX IF NOT EXISTS project_schedules_project_id_idx ON project_schedules(project_id);
CREATE INDEX IF NOT EXISTS project_schedules_status_idx ON project_schedules(status);
CREATE INDEX IF NOT EXISTS project_schedules_priority_idx ON project_schedules(priority);
CREATE INDEX IF NOT EXISTS project_schedules_assigned_to_idx ON project_schedules(assigned_to);
CREATE INDEX IF NOT EXISTS project_schedules_planned_start_idx ON project_schedules(planned_start);
CREATE INDEX IF NOT EXISTS project_schedules_planned_end_idx ON project_schedules(planned_end);
CREATE INDEX IF NOT EXISTS project_schedules_actual_start_idx ON project_schedules(actual_start);
CREATE INDEX IF NOT EXISTS project_schedules_actual_end_idx ON project_schedules(actual_end);
CREATE INDEX IF NOT EXISTS project_schedules_progress_idx ON project_schedules(progress_percentage);
CREATE INDEX IF NOT EXISTS project_schedules_schedule_code_idx ON project_schedules(schedule_code);
CREATE INDEX IF NOT EXISTS project_schedules_baseline_start_idx ON project_schedules(baseline_start);
CREATE INDEX IF NOT EXISTS project_schedules_baseline_end_idx ON project_schedules(baseline_end);
CREATE INDEX IF NOT EXISTS project_schedules_vendor_id_idx ON project_schedules(vendor_id);
CREATE INDEX IF NOT EXISTS project_schedules_milestone_idx ON project_schedules(milestone);
CREATE INDEX IF NOT EXISTS project_schedules_schedule_source_idx ON project_schedules(schedule_source);
CREATE INDEX IF NOT EXISTS project_schedules_created_by_idx ON project_schedules(created_by);
CREATE INDEX IF NOT EXISTS project_schedules_crew_size_idx ON project_schedules(crew_size);
CREATE INDEX IF NOT EXISTS project_schedules_duration_days_idx ON project_schedules(duration_days);
CREATE INDEX IF NOT EXISTS project_schedules_activity_category_idx ON project_schedules(activity_category);

-- Indexes for project_schedule_dependencies table
CREATE INDEX IF NOT EXISTS project_schedule_dependencies_workspace_id_idx ON project_schedule_dependencies(workspace_id);
CREATE INDEX IF NOT EXISTS project_schedule_dependencies_project_id_idx ON project_schedule_dependencies(project_id);
CREATE INDEX IF NOT EXISTS project_schedule_dependencies_schedule_id_idx ON project_schedule_dependencies(schedule_id);
CREATE INDEX IF NOT EXISTS project_schedule_dependencies_depends_on_idx ON project_schedule_dependencies(depends_on_schedule_id);
CREATE INDEX IF NOT EXISTS project_schedule_dependencies_type_idx ON project_schedule_dependencies(type);

-- Indexes for new project management tables
CREATE INDEX IF NOT EXISTS project_budgets_workspace_id_idx ON project_budgets(workspace_id);
CREATE INDEX IF NOT EXISTS project_budgets_project_id_idx ON project_budgets(project_id);
CREATE INDEX IF NOT EXISTS project_budgets_category_idx ON project_budgets(category);
CREATE INDEX IF NOT EXISTS project_budgets_contact_id_idx ON project_budgets(contact_id);
CREATE INDEX IF NOT EXISTS project_budgets_vendor_id_idx ON project_budgets(vendor_id);
CREATE INDEX IF NOT EXISTS project_budgets_status_idx ON project_budgets(status);
CREATE INDEX IF NOT EXISTS project_budgets_budget_type_idx ON project_budgets(budget_type);

CREATE INDEX IF NOT EXISTS material_tracking_workspace_id_idx ON material_tracking(workspace_id);
CREATE INDEX IF NOT EXISTS material_tracking_project_id_idx ON material_tracking(project_id);
CREATE INDEX IF NOT EXISTS material_tracking_vendor_id_idx ON material_tracking(vendor_id);
CREATE INDEX IF NOT EXISTS material_tracking_status_idx ON material_tracking(status);
CREATE INDEX IF NOT EXISTS material_tracking_delivery_date_idx ON material_tracking(delivery_date);

CREATE INDEX IF NOT EXISTS equipment_tracking_workspace_id_idx ON equipment_tracking(workspace_id);
CREATE INDEX IF NOT EXISTS equipment_tracking_project_id_idx ON equipment_tracking(project_id);
CREATE INDEX IF NOT EXISTS equipment_tracking_vendor_id_idx ON equipment_tracking(vendor_id);
CREATE INDEX IF NOT EXISTS equipment_tracking_status_idx ON equipment_tracking(status);
CREATE INDEX IF NOT EXISTS equipment_tracking_usage_date_idx ON equipment_tracking(usage_date);

CREATE INDEX IF NOT EXISTS labor_tracking_workspace_id_idx ON labor_tracking(workspace_id);
CREATE INDEX IF NOT EXISTS labor_tracking_project_id_idx ON labor_tracking(project_id);
CREATE INDEX IF NOT EXISTS labor_tracking_worker_id_idx ON labor_tracking(worker_id);
CREATE INDEX IF NOT EXISTS labor_tracking_worker_name_idx ON labor_tracking(worker_name);
CREATE INDEX IF NOT EXISTS labor_tracking_work_date_idx ON labor_tracking(work_date);
CREATE INDEX IF NOT EXISTS labor_tracking_activity_idx ON labor_tracking(activity);

CREATE INDEX IF NOT EXISTS vendors_workspace_id_idx ON vendors(workspace_id);
CREATE INDEX IF NOT EXISTS vendors_project_id_idx ON vendors(project_id);
CREATE INDEX IF NOT EXISTS vendors_type_idx ON vendors(vendor_type);

CREATE INDEX IF NOT EXISTS vendor_rates_vendor_id_idx ON vendor_rates(vendor_id);
CREATE INDEX IF NOT EXISTS vendor_rates_workspace_id_idx ON vendor_rates(workspace_id);
CREATE INDEX IF NOT EXISTS vendor_rates_project_id_idx ON vendor_rates(project_id);
CREATE INDEX IF NOT EXISTS vendor_rates_rate_type_idx ON vendor_rates(rate_type);
CREATE INDEX IF NOT EXISTS vendor_rates_effective_from_idx ON vendor_rates(effective_from);
CREATE INDEX IF NOT EXISTS vendor_rates_is_active_idx ON vendor_rates(is_active);

-- Indexes for pricing and alerts tables
CREATE INDEX IF NOT EXISTS pricing_plans_name_idx ON pricing_plans(name);
CREATE INDEX IF NOT EXISTS pricing_plans_is_active_idx ON pricing_plans(is_active);
CREATE INDEX IF NOT EXISTS pricing_plans_min_size_idx ON pricing_plans(min_project_size);
CREATE INDEX IF NOT EXISTS pricing_plans_max_size_idx ON pricing_plans(max_project_size);

CREATE INDEX IF NOT EXISTS project_alerts_workspace_id_idx ON project_alerts(workspace_id);
CREATE INDEX IF NOT EXISTS project_alerts_project_id_idx ON project_alerts(project_id);
CREATE INDEX IF NOT EXISTS project_alerts_alert_type_idx ON project_alerts(alert_type);
CREATE INDEX IF NOT EXISTS project_alerts_status_idx ON project_alerts(status);
CREATE INDEX IF NOT EXISTS project_alerts_severity_idx ON project_alerts(severity);
CREATE INDEX IF NOT EXISTS project_alerts_context_idx ON project_alerts(context_table, context_id);
CREATE INDEX IF NOT EXISTS project_alerts_metadata_idx ON project_alerts USING GIN(metadata);
CREATE INDEX IF NOT EXISTS project_alerts_created_at_idx ON project_alerts(created_at);
CREATE INDEX IF NOT EXISTS project_alerts_resolved_at_idx ON project_alerts(resolved_at);
CREATE INDEX IF NOT EXISTS project_alerts_resolved_by_idx ON project_alerts(resolved_by);

-- Indexes for project_usage table
CREATE INDEX IF NOT EXISTS project_usage_workspace_id_idx ON project_usage(workspace_id);
CREATE INDEX IF NOT EXISTS project_usage_project_id_idx ON project_usage(project_id);
CREATE INDEX IF NOT EXISTS project_usage_call_minutes_idx ON project_usage(call_minutes_used);
CREATE INDEX IF NOT EXISTS project_usage_tokens_idx ON project_usage(tokens_used);

-- Indexes for AI recommendations table
CREATE INDEX IF NOT EXISTS ai_recommendations_workspace_id_idx ON ai_recommendations(workspace_id);
CREATE INDEX IF NOT EXISTS ai_recommendations_project_id_idx ON ai_recommendations(project_id);
CREATE INDEX IF NOT EXISTS ai_recommendations_type_idx ON ai_recommendations(recommendation_type);
CREATE INDEX IF NOT EXISTS ai_recommendations_status_idx ON ai_recommendations(status);
CREATE INDEX IF NOT EXISTS ai_recommendations_reviewed_by_idx ON ai_recommendations(reviewed_by);
CREATE INDEX IF NOT EXISTS ai_recommendations_generated_at_idx ON ai_recommendations(generated_at);
CREATE INDEX IF NOT EXISTS ai_recommendations_context_idx ON ai_recommendations(context_table, context_id);
CREATE INDEX IF NOT EXISTS ai_recommendations_details_idx ON ai_recommendations USING GIN(details);

-- Indexes for project_delays table
CREATE INDEX IF NOT EXISTS project_delays_workspace_id_idx ON project_delays(workspace_id);
CREATE INDEX IF NOT EXISTS project_delays_project_id_idx ON project_delays(project_id);
CREATE INDEX IF NOT EXISTS project_delays_delay_date_idx ON project_delays(delay_date);
CREATE INDEX IF NOT EXISTS project_delays_delay_type_idx ON project_delays(delay_type);
CREATE INDEX IF NOT EXISTS project_delays_delay_category_idx ON project_delays(delay_category);
CREATE INDEX IF NOT EXISTS project_delays_status_idx ON project_delays(status);
CREATE INDEX IF NOT EXISTS project_delays_priority_idx ON project_delays(priority);
CREATE INDEX IF NOT EXISTS project_delays_responsible_party_name_idx ON project_delays(responsible_party_name);

-- Indexes for project_rfis table
CREATE INDEX IF NOT EXISTS project_rfis_workspace_id_idx ON project_rfis(workspace_id);
CREATE INDEX IF NOT EXISTS project_rfis_project_id_idx ON project_rfis(project_id);
CREATE INDEX IF NOT EXISTS project_rfis_rfi_date_idx ON project_rfis(rfi_date);
CREATE INDEX IF NOT EXISTS project_rfis_rfi_type_idx ON project_rfis(rfi_type);
CREATE INDEX IF NOT EXISTS project_rfis_status_idx ON project_rfis(status);
CREATE INDEX IF NOT EXISTS project_rfis_priority_idx ON project_rfis(priority);
CREATE INDEX IF NOT EXISTS project_rfis_requested_by_idx ON project_rfis(requested_by);
CREATE INDEX IF NOT EXISTS project_rfis_assigned_to_idx ON project_rfis(assigned_to);
CREATE INDEX IF NOT EXISTS project_rfis_due_date_idx ON project_rfis(due_date);

-- Indexes for project_safety_incidents table
CREATE INDEX IF NOT EXISTS project_safety_incidents_workspace_id_idx ON project_safety_incidents(workspace_id);
CREATE INDEX IF NOT EXISTS project_safety_incidents_project_id_idx ON project_safety_incidents(project_id);
CREATE INDEX IF NOT EXISTS project_safety_incidents_incident_date_idx ON project_safety_incidents(incident_date);
CREATE INDEX IF NOT EXISTS project_safety_incidents_incident_type_idx ON project_safety_incidents(incident_type);
CREATE INDEX IF NOT EXISTS project_safety_incidents_severity_idx ON project_safety_incidents(severity);
CREATE INDEX IF NOT EXISTS project_safety_incidents_status_idx ON project_safety_incidents(status);
CREATE INDEX IF NOT EXISTS project_safety_incidents_reported_by_idx ON project_safety_incidents(reported_by);
CREATE INDEX IF NOT EXISTS project_safety_incidents_assigned_to_idx ON project_safety_incidents(assigned_to);

-- Indexes for project_actions table
CREATE INDEX IF NOT EXISTS project_actions_workspace_id_idx ON project_actions(workspace_id);
CREATE INDEX IF NOT EXISTS project_actions_project_id_idx ON project_actions(project_id);
CREATE INDEX IF NOT EXISTS project_actions_action_item_date_idx ON project_actions(action_item_date);
CREATE INDEX IF NOT EXISTS project_actions_status_idx ON project_actions(status);
CREATE INDEX IF NOT EXISTS project_actions_priority_idx ON project_actions(priority);
CREATE INDEX IF NOT EXISTS project_actions_assigned_to_idx ON project_actions(assigned_to);
CREATE INDEX IF NOT EXISTS project_actions_deadline_idx ON project_actions(deadline);
CREATE INDEX IF NOT EXISTS project_actions_ai_recommendation_id_idx ON project_actions(ai_recommendation_id);
CREATE INDEX IF NOT EXISTS project_actions_task_id_idx ON project_actions(task_id);

-- Indexes for project_issues table
CREATE INDEX IF NOT EXISTS project_issues_workspace_id_idx ON project_issues(workspace_id);
CREATE INDEX IF NOT EXISTS project_issues_project_id_idx ON project_issues(project_id);
CREATE INDEX IF NOT EXISTS project_issues_issue_date_idx ON project_issues(issue_date);
CREATE INDEX IF NOT EXISTS project_issues_severity_idx ON project_issues(severity);
CREATE INDEX IF NOT EXISTS project_issues_status_idx ON project_issues(status);
CREATE INDEX IF NOT EXISTS project_issues_assigned_to_idx ON project_issues(assigned_to);
CREATE INDEX IF NOT EXISTS project_issues_immediate_action_needed_idx ON project_issues(immediate_action_needed);

-- Indexes for project_work_progress table
CREATE INDEX IF NOT EXISTS project_work_progress_workspace_id_idx ON project_work_progress(workspace_id);
CREATE INDEX IF NOT EXISTS project_work_progress_project_id_idx ON project_work_progress(project_id);
CREATE INDEX IF NOT EXISTS project_work_progress_call_uuid_idx ON project_work_progress(call_uuid);
CREATE INDEX IF NOT EXISTS project_work_progress_activity_date_idx ON project_work_progress(activity_date);
CREATE INDEX IF NOT EXISTS project_work_progress_activity_category_idx ON project_work_progress(activity_category);
CREATE INDEX IF NOT EXISTS project_work_progress_progress_percentage_idx ON project_work_progress(progress_percentage);
CREATE INDEX IF NOT EXISTS project_work_progress_status_idx ON project_work_progress(status);

-- Indexes for work_progress table
CREATE INDEX IF NOT EXISTS work_progress_workspace_id_idx ON work_progress(workspace_id);
CREATE INDEX IF NOT EXISTS work_progress_project_id_idx ON work_progress(project_id);
CREATE INDEX IF NOT EXISTS work_progress_activity_date_idx ON work_progress(activity_date);
CREATE INDEX IF NOT EXISTS work_progress_activity_category_idx ON work_progress(activity_category);
CREATE INDEX IF NOT EXISTS work_progress_reported_by_idx ON work_progress(reported_by);
CREATE INDEX IF NOT EXISTS work_progress_status_idx ON work_progress(status);
CREATE INDEX IF NOT EXISTS work_progress_materials_consumed_idx ON work_progress USING GIN(materials_consumed);
CREATE INDEX IF NOT EXISTS work_progress_equipment_used_idx ON work_progress USING GIN(equipment_used);
CREATE INDEX IF NOT EXISTS work_progress_labor_employed_idx ON work_progress USING GIN(labor_employed);

-- Indexes for issues table
CREATE INDEX IF NOT EXISTS issues_workspace_id_idx ON issues(workspace_id);
CREATE INDEX IF NOT EXISTS issues_project_id_idx ON issues(project_id);
CREATE INDEX IF NOT EXISTS issues_tag_idx ON issues(tag);
CREATE INDEX IF NOT EXISTS issues_status_idx ON issues(status);
CREATE INDEX IF NOT EXISTS issues_reported_by_idx ON issues(reported_by);
CREATE INDEX IF NOT EXISTS issues_need_by_idx ON issues(need_by);
CREATE INDEX IF NOT EXISTS issues_due_time_idx ON issues(due_time);
CREATE INDEX IF NOT EXISTS issues_priority_idx ON issues(priority);
CREATE INDEX IF NOT EXISTS issues_tag_data_idx ON issues USING GIN(tag_data);

-- Indexes for site_media table
CREATE INDEX IF NOT EXISTS site_media_workspace_id_idx ON site_media(workspace_id);
CREATE INDEX IF NOT EXISTS site_media_project_id_idx ON site_media(project_id);
CREATE INDEX IF NOT EXISTS site_media_file_path_idx ON site_media(file_path);
CREATE INDEX IF NOT EXISTS site_media_filename_idx ON site_media(filename);
CREATE INDEX IF NOT EXISTS site_media_capture_date_idx ON site_media(capture_date);
CREATE INDEX IF NOT EXISTS site_media_media_category_idx ON site_media(media_category);
CREATE INDEX IF NOT EXISTS site_media_uploaded_by_idx ON site_media(uploaded_by);
CREATE INDEX IF NOT EXISTS site_media_tags_idx ON site_media USING GIN(tags);

-- Indexes for project_cashflows table
CREATE INDEX IF NOT EXISTS project_cashflows_workspace_id_idx ON project_cashflows(workspace_id);
CREATE INDEX IF NOT EXISTS project_cashflows_project_id_idx ON project_cashflows(project_id);
CREATE INDEX IF NOT EXISTS project_cashflows_budget_id_idx ON project_cashflows(budget_id);
CREATE INDEX IF NOT EXISTS project_cashflows_contact_id_idx ON project_cashflows(contact_id);
CREATE INDEX IF NOT EXISTS project_cashflows_user_id_idx ON project_cashflows(user_id);
CREATE INDEX IF NOT EXISTS project_cashflows_flow_type_idx ON project_cashflows(flow_type);
CREATE INDEX IF NOT EXISTS project_cashflows_amount_idx ON project_cashflows(amount);
CREATE INDEX IF NOT EXISTS project_cashflows_category_idx ON project_cashflows(category);
CREATE INDEX IF NOT EXISTS project_cashflows_payment_mode_idx ON project_cashflows(payment_mode);
CREATE INDEX IF NOT EXISTS project_cashflows_reference_no_idx ON project_cashflows(reference_no);
CREATE INDEX IF NOT EXISTS project_cashflows_created_at_idx ON project_cashflows(created_at);

-- Indexes for sla_settings table
CREATE INDEX IF NOT EXISTS sla_settings_workspace_id_idx ON sla_settings(workspace_id);
CREATE INDEX IF NOT EXISTS sla_settings_project_id_idx ON sla_settings(project_id);
CREATE INDEX IF NOT EXISTS sla_settings_entity_type_idx ON sla_settings(entity_type);
CREATE INDEX IF NOT EXISTS sla_settings_entity_subtype_idx ON sla_settings(entity_subtype);
CREATE INDEX IF NOT EXISTS sla_settings_trigger_name_idx ON sla_settings(trigger_name);
CREATE INDEX IF NOT EXISTS sla_settings_trigger_condition_idx ON sla_settings USING GIN(trigger_condition);
CREATE INDEX IF NOT EXISTS sla_settings_ai_action_enabled_idx ON sla_settings(ai_action_enabled);

-- Indexes for subtasks table
CREATE INDEX IF NOT EXISTS subtasks_workspace_id_idx ON subtasks(workspace_id);
CREATE INDEX IF NOT EXISTS subtasks_project_id_idx ON subtasks(project_id);
CREATE INDEX IF NOT EXISTS subtasks_task_id_idx ON subtasks(task_id);
CREATE INDEX IF NOT EXISTS subtasks_issue_id_idx ON subtasks(issue_id);
CREATE INDEX IF NOT EXISTS subtasks_assigned_to_idx ON subtasks(assigned_to);
CREATE INDEX IF NOT EXISTS subtasks_status_idx ON subtasks(status);
CREATE INDEX IF NOT EXISTS subtasks_sequence_number_idx ON subtasks(sequence_number);
CREATE INDEX IF NOT EXISTS subtasks_due_time_idx ON subtasks(due_time);

-- Indexes for project_member_access table
CREATE INDEX IF NOT EXISTS project_member_access_workspace_id_idx ON project_member_access(workspace_id);
CREATE INDEX IF NOT EXISTS project_member_access_project_id_idx ON project_member_access(project_id);
CREATE INDEX IF NOT EXISTS project_member_access_user_id_idx ON project_member_access(user_id);
CREATE INDEX IF NOT EXISTS project_member_access_access_level_idx ON project_member_access(access_level);
CREATE INDEX IF NOT EXISTS project_member_access_granted_by_idx ON project_member_access(granted_by);

-- Indexes for dpr_call_settings table
CREATE INDEX IF NOT EXISTS dpr_call_settings_workspace_id_idx ON dpr_call_settings(workspace_id);
CREATE INDEX IF NOT EXISTS dpr_call_settings_project_id_idx ON dpr_call_settings(project_id);
CREATE INDEX IF NOT EXISTS dpr_call_settings_project_contact_id_idx ON dpr_call_settings(project_contact_id);
CREATE INDEX IF NOT EXISTS dpr_call_settings_enabled_idx ON dpr_call_settings(enabled);
CREATE INDEX IF NOT EXISTS dpr_call_settings_from_date_idx ON dpr_call_settings(from_date);
CREATE INDEX IF NOT EXISTS dpr_call_settings_to_date_idx ON dpr_call_settings(to_date);

-- Enable Row Level Security
ALTER TABLE workspaces ENABLE ROW LEVEL SECURITY;
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE workspace_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE agents ENABLE ROW LEVEL SECURITY;
ALTER TABLE plivo_subaccounts ENABLE ROW LEVEL SECURITY;
ALTER TABLE plivo_endpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE phone_numbers ENABLE ROW LEVEL SECURITY;
ALTER TABLE phone_pools ENABLE ROW LEVEL SECURITY;
ALTER TABLE phone_pool_memberships ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_phone_mappings ENABLE ROW LEVEL SECURITY;

ALTER TABLE call_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE contacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE kb_files ENABLE ROW LEVEL SECURITY;
ALTER TABLE call_analysis_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE project_schedules ENABLE ROW LEVEL SECURITY;
ALTER TABLE project_schedule_dependencies ENABLE ROW LEVEL SECURITY;
ALTER TABLE project_budgets ENABLE ROW LEVEL SECURITY;
ALTER TABLE material_tracking ENABLE ROW LEVEL SECURITY;
ALTER TABLE equipment_tracking ENABLE ROW LEVEL SECURITY;
ALTER TABLE labor_tracking ENABLE ROW LEVEL SECURITY;
ALTER TABLE vendors ENABLE ROW LEVEL SECURITY;
ALTER TABLE vendor_rates ENABLE ROW LEVEL SECURITY;
ALTER TABLE pricing_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE project_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_recommendations ENABLE ROW LEVEL SECURITY;
ALTER TABLE project_usage ENABLE ROW LEVEL SECURITY;
ALTER TABLE project_delays ENABLE ROW LEVEL SECURITY;
ALTER TABLE project_rfis ENABLE ROW LEVEL SECURITY;
ALTER TABLE project_safety_incidents ENABLE ROW LEVEL SECURITY;
ALTER TABLE project_actions ENABLE ROW LEVEL SECURITY;
ALTER TABLE project_issues ENABLE ROW LEVEL SECURITY;
ALTER TABLE project_work_progress ENABLE ROW LEVEL SECURITY;
ALTER TABLE work_progress ENABLE ROW LEVEL SECURITY;
ALTER TABLE project_contacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE dpr_call_settings ENABLE ROW LEVEL SECURITY;
ALTER TABLE issues ENABLE ROW LEVEL SECURITY;
ALTER TABLE site_media ENABLE ROW LEVEL SECURITY;
ALTER TABLE project_member_access ENABLE ROW LEVEL SECURITY;
ALTER TABLE sla_settings ENABLE ROW LEVEL SECURITY;
ALTER TABLE subtasks ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for workspaces
CREATE POLICY "Users can view their workspaces"
    ON workspaces FOR SELECT
    TO authenticated
    USING (
        id IN (
            SELECT workspace_id FROM workspace_members
            WHERE user_id = (select auth.uid())
        )
    );

CREATE POLICY "Users can update their workspaces"
    ON workspaces FOR UPDATE
    TO authenticated
    USING (owner_id = (select auth.uid()));

-- Create RLS policies for profiles
CREATE POLICY "Users can view their own profile"
    ON profiles FOR SELECT
    TO authenticated
    USING (user_id = (select auth.uid()));

CREATE POLICY "Users can insert their own profile"
    ON profiles FOR INSERT
    TO authenticated
    WITH CHECK (user_id = (select auth.uid()));

CREATE POLICY "Users can update their own profile"
    ON profiles FOR UPDATE
    TO authenticated
    USING (user_id = (select auth.uid()));

-- Create RLS policies for workspace_members (Insert/ Update managed through backend)
CREATE POLICY "Users can view their own workspace memberships"
    ON workspace_members FOR SELECT
    TO authenticated
    USING (user_id = (select auth.uid()));


-- Create RLS policies for agents
CREATE POLICY "Owners/ Admins can view/manage agents in their workspace"
    ON agents FOR ALL
    TO authenticated
    USING (
        workspace_id IN (
            SELECT workspace_id FROM workspace_members
            WHERE user_id = (select auth.uid()) AND role IN ('owner', 'admin')
        )
    );

-- Create RLS policies for phone_numbers
CREATE POLICY "Owners/ Admins can manage phone numbers in their workspace"
    ON phone_numbers FOR ALL
    TO authenticated
    USING (
        workspace_id IN (
            SELECT workspace_id FROM workspace_members
            WHERE user_id = (select auth.uid()) AND role IN ('owner', 'admin')
        )
    );

-- Create RLS policies for phone_pools

CREATE POLICY "Owners/ Admins can manage phone pools in their workspace"
    ON phone_pools FOR ALL
    TO authenticated
    USING (
        workspace_id IN (
            SELECT workspace_id FROM workspace_members
            WHERE user_id = (select auth.uid()) AND role IN ('owner', 'admin')
        )
    );

-- Create RLS policies for phone_pool_memberships

CREATE POLICY "Owners/ Admins can manage phone pool memberships in their workspace"
    ON phone_pool_memberships FOR ALL
    TO authenticated
    USING (
        phone_pool_id IN (
            SELECT id FROM phone_pools
            WHERE workspace_id IN (
                SELECT workspace_id FROM workspace_members
                WHERE user_id = (select auth.uid()) AND role IN ('owner', 'admin')
            )
        )
    );

-- Create RLS policies for agent_phone_mappings

CREATE POLICY "Owners/ Admins can manage agent phone mappings in their workspace"
    ON agent_phone_mappings FOR ALL
    TO authenticated
    USING (
        workspace_id IN (
            SELECT workspace_id FROM workspace_members
            WHERE user_id = (select auth.uid()) AND role IN ('owner', 'admin')
        )
    );



-- Create RLS policies for call_logs
CREATE POLICY "Owners/ Admins can view call metrics in their workspace"
    ON call_logs FOR SELECT
    TO authenticated
    USING (
        workspace_id IN (
            SELECT workspace_id FROM workspace_members
            WHERE user_id = (select auth.uid()) AND role IN ('owner', 'admin')
        )
    );

-- Create RLS policies for tasks

CREATE POLICY "Owners/ Admins can manage tasks in their workspace"
    ON tasks FOR ALL
    TO authenticated
    USING (
        workspace_id IN (
            SELECT workspace_id FROM workspace_members
            WHERE user_id = (select auth.uid()) AND role IN ('owner', 'admin')
        )
    );

-- Create RLS policies for contacts

CREATE POLICY "User can view contacts in their workspace"
    ON contacts FOR SELECT
    TO authenticated
    USING (
        workspace_id IN (
            SELECT workspace_id FROM workspace_members
            WHERE user_id = (select auth.uid())
        )
    );

CREATE POLICY "Owners/ Admins can manage contacts in their workspace"
    ON contacts FOR ALL
    TO authenticated
    USING (
        workspace_id IN (
            SELECT workspace_id FROM workspace_members
            WHERE user_id = (select auth.uid()) AND role IN ('owner', 'admin')
        )
    );

-- RLS Policies for projects table
-- Policy: Workspace owners/admins have full access to all projects in their workspaces
CREATE POLICY "Projects - Owner and admin access" ON projects
FOR ALL TO authenticated
USING (
  workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
);

-- Policy: Members can view projects based on project access level
CREATE POLICY "Projects - Member view access" ON projects
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = projects.workspace_id
    AND pma.project_id = projects.id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level IN ('view', 'edit')
  )
);

-- Policy: Members can insert projects if they have edit access to the project
CREATE POLICY "Projects - Member insert access" ON projects
FOR INSERT TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = projects.workspace_id
    AND pma.project_id = projects.id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can update projects if they have edit access to the project
CREATE POLICY "Projects - Member update access" ON projects
FOR UPDATE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = projects.workspace_id
    AND pma.project_id = projects.id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can delete projects if they have edit access to the project
CREATE POLICY "Projects - Member delete access" ON projects
FOR DELETE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = projects.workspace_id
    AND pma.project_id = projects.id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- RLS Policies for project_schedules table
-- Policy: Workspace owners/admins have full access to all project schedules in their workspaces
CREATE POLICY "Project Schedules - Owner and admin access" ON project_schedules
FOR ALL TO authenticated
USING (
  workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
);

-- Policy: Members can view project schedules based on project access level
CREATE POLICY "Project Schedules - Member view access" ON project_schedules
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_schedules.workspace_id
    AND pma.project_id = project_schedules.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level IN ('view', 'edit')
  )
);

-- Policy: Members can insert project schedules if they have edit access to the project
CREATE POLICY "Project Schedules - Member insert access" ON project_schedules
FOR INSERT TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_schedules.workspace_id
    AND pma.project_id = project_schedules.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can update project schedules if they have edit access to the project
CREATE POLICY "Project Schedules - Member update access" ON project_schedules
FOR UPDATE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_schedules.workspace_id
    AND pma.project_id = project_schedules.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can delete project schedules if they have edit access to the project
CREATE POLICY "Project Schedules - Member delete access" ON project_schedules
FOR DELETE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_schedules.workspace_id
    AND pma.project_id = project_schedules.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- RLS Policies for project_schedule_dependencies table
-- Policy: Workspace owners/admins have full access to all project schedule dependencies in their workspaces
CREATE POLICY "Project Schedule Dependencies - Owner and admin access" ON project_schedule_dependencies
FOR ALL TO authenticated
USING (
  workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
);

-- Policy: Members can view project schedule dependencies based on project access level
CREATE POLICY "Project Schedule Dependencies - Member view access" ON project_schedule_dependencies
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_schedule_dependencies.workspace_id
    AND pma.project_id = project_schedule_dependencies.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level IN ('view', 'edit')
  )
);

-- Policy: Members can insert project schedule dependencies if they have edit access to the project
CREATE POLICY "Project Schedule Dependencies - Member insert access" ON project_schedule_dependencies
FOR INSERT TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_schedule_dependencies.workspace_id
    AND pma.project_id = project_schedule_dependencies.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can update project schedule dependencies if they have edit access to the project
CREATE POLICY "Project Schedule Dependencies - Member update access" ON project_schedule_dependencies
FOR UPDATE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_schedule_dependencies.workspace_id
    AND pma.project_id = project_schedule_dependencies.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can delete project schedule dependencies if they have edit access to the project
CREATE POLICY "Project Schedule Dependencies - Member delete access" ON project_schedule_dependencies
FOR DELETE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_schedule_dependencies.workspace_id
    AND pma.project_id = project_schedule_dependencies.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Create RLS policies for project_budgets (restricted to owners and admins only)
CREATE POLICY "Only workspace owners and admins can access project budgets"
    ON project_budgets FOR ALL
    TO authenticated
    USING (
        workspace_id IN (
            SELECT workspace_id FROM workspace_members
            WHERE user_id = (select auth.uid()) AND role IN ('owner', 'admin')
        )
    );

-- RLS Policies for material_tracking table
-- Policy: Workspace owners/admins have full access to all material tracking in their workspaces
CREATE POLICY "Material Tracking - Owner and admin access" ON material_tracking
FOR ALL TO authenticated
USING (
  workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
);

-- Policy: Members can view material tracking based on project access level
CREATE POLICY "Material Tracking - Member view access" ON material_tracking
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = material_tracking.workspace_id
    AND pma.project_id = material_tracking.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level IN ('view', 'edit')
  )
);

-- Policy: Members can insert material tracking if they have edit access to the project
CREATE POLICY "Material Tracking - Member insert access" ON material_tracking
FOR INSERT TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = material_tracking.workspace_id
    AND pma.project_id = material_tracking.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can update material tracking if they have edit access to the project
CREATE POLICY "Material Tracking - Member update access" ON material_tracking
FOR UPDATE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = material_tracking.workspace_id
    AND pma.project_id = material_tracking.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can delete material tracking if they have edit access to the project
CREATE POLICY "Material Tracking - Member delete access" ON material_tracking
FOR DELETE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = material_tracking.workspace_id
    AND pma.project_id = material_tracking.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- RLS Policies for equipment_tracking table
-- Policy: Workspace owners/admins have full access to all equipment tracking in their workspaces
CREATE POLICY "Equipment Tracking - Owner and admin access" ON equipment_tracking
FOR ALL TO authenticated
USING (
  workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
);

-- Policy: Members can view equipment tracking based on project access level
CREATE POLICY "Equipment Tracking - Member view access" ON equipment_tracking
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = equipment_tracking.workspace_id
    AND pma.project_id = equipment_tracking.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level IN ('view', 'edit')
  )
);

-- Policy: Members can insert equipment tracking if they have edit access to the project
CREATE POLICY "Equipment Tracking - Member insert access" ON equipment_tracking
FOR INSERT TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = equipment_tracking.workspace_id
    AND pma.project_id = equipment_tracking.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can update equipment tracking if they have edit access to the project
CREATE POLICY "Equipment Tracking - Member update access" ON equipment_tracking
FOR UPDATE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = equipment_tracking.workspace_id
    AND pma.project_id = equipment_tracking.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can delete equipment tracking if they have edit access to the project
CREATE POLICY "Equipment Tracking - Member delete access" ON equipment_tracking
FOR DELETE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = equipment_tracking.workspace_id
    AND pma.project_id = equipment_tracking.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- RLS Policies for labor_tracking table
-- Policy: Workspace owners/admins have full access to all labor tracking in their workspaces
CREATE POLICY "Labor Tracking - Owner and admin access" ON labor_tracking
FOR ALL TO authenticated
USING (
  workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
);

-- Policy: Members can view labor tracking based on project access level
CREATE POLICY "Labor Tracking - Member view access" ON labor_tracking
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = labor_tracking.workspace_id
    AND pma.project_id = labor_tracking.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level IN ('view', 'edit')
  )
);

-- Policy: Members can insert labor tracking if they have edit access to the project
CREATE POLICY "Labor Tracking - Member insert access" ON labor_tracking
FOR INSERT TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = labor_tracking.workspace_id
    AND pma.project_id = labor_tracking.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can update labor tracking if they have edit access to the project
CREATE POLICY "Labor Tracking - Member update access" ON labor_tracking
FOR UPDATE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = labor_tracking.workspace_id
    AND pma.project_id = labor_tracking.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can delete labor tracking if they have edit access to the project
CREATE POLICY "Labor Tracking - Member delete access" ON labor_tracking
FOR DELETE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = labor_tracking.workspace_id
    AND pma.project_id = labor_tracking.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Create RLS policies for vendors

CREATE POLICY "Owners/ Admins can manage vendors in their workspace"
    ON vendors FOR ALL
    TO authenticated
    USING (
        workspace_id IN (
            SELECT workspace_id FROM workspace_members
            WHERE user_id = (select auth.uid()) AND role IN ('owner', 'admin')
        )
    );

-- Create RLS policies for vendor_rates

CREATE POLICY "Owners/ Admins can manage vendor rates in their workspace"
    ON vendor_rates FOR ALL
    TO authenticated
    USING (
        workspace_id IN (
            SELECT workspace_id FROM workspace_members
            WHERE user_id = (select auth.uid()) AND role IN ('owner', 'admin')
        )
    );

-- RLS Policies for project_alerts table
-- Policy: Workspace owners/admins have full access to all project alerts in their workspaces
CREATE POLICY "Project Alerts - Owner and admin access" ON project_alerts
FOR ALL TO authenticated
USING (
  workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
);

-- Policy: Members can view project alerts based on project access level
CREATE POLICY "Project Alerts - Member view access" ON project_alerts
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_alerts.workspace_id
    AND pma.project_id = project_alerts.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level IN ('view', 'edit')
  )
);

-- Policy: Members can insert project alerts if they have edit access to the project
CREATE POLICY "Project Alerts - Member insert access" ON project_alerts
FOR INSERT TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_alerts.workspace_id
    AND pma.project_id = project_alerts.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can update project alerts if they have edit access to the project
CREATE POLICY "Project Alerts - Member update access" ON project_alerts
FOR UPDATE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_alerts.workspace_id
    AND pma.project_id = project_alerts.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can delete project alerts if they have edit access to the project
CREATE POLICY "Project Alerts - Member delete access" ON project_alerts
FOR DELETE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_alerts.workspace_id
    AND pma.project_id = project_alerts.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- RLS Policies for ai_recommendations table
-- Policy: Workspace owners/admins have full access to all AI recommendations in their workspaces
CREATE POLICY "AI Recommendations - Owner and admin access" ON ai_recommendations
FOR ALL TO authenticated
USING (
  workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
);

-- Policy: Members can view AI recommendations based on project access level
CREATE POLICY "AI Recommendations - Member view access" ON ai_recommendations
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = ai_recommendations.workspace_id
    AND pma.project_id = ai_recommendations.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level IN ('view', 'edit')
  )
);

-- Policy: Members can insert AI recommendations if they have edit access to the project
CREATE POLICY "AI Recommendations - Member insert access" ON ai_recommendations
FOR INSERT TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = ai_recommendations.workspace_id
    AND pma.project_id = ai_recommendations.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can update AI recommendations if they have edit access to the project
CREATE POLICY "AI Recommendations - Member update access" ON ai_recommendations
FOR UPDATE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = ai_recommendations.workspace_id
    AND pma.project_id = ai_recommendations.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can delete AI recommendations if they have edit access to the project
CREATE POLICY "AI Recommendations - Member delete access" ON ai_recommendations
FOR DELETE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = ai_recommendations.workspace_id
    AND pma.project_id = ai_recommendations.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Create RLS policies for project_usage
CREATE POLICY "Owners/ Admins can view project usage in their workspace"
    ON project_usage FOR SELECT
    TO authenticated
    USING (
        workspace_id IN (
            SELECT workspace_id FROM workspace_members
            WHERE user_id = (select auth.uid()) AND role IN ('owner', 'admin')
        )
    );

-- RLS Policies for kb_files table
-- Policy: Workspace owners/admins have full access to all KB files in their workspaces
CREATE POLICY "KB Files - Owner and admin access" ON kb_files
FOR ALL TO authenticated
USING (
  workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
);

-- Policy: Members can view KB files based on project access level
CREATE POLICY "KB Files - Member view access" ON kb_files
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = kb_files.workspace_id
    AND pma.project_id = kb_files.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level IN ('view', 'edit')
  )
);

-- Policy: Members can insert KB files if they have edit access to the project
CREATE POLICY "KB Files - Member insert access" ON kb_files
FOR INSERT TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = kb_files.workspace_id
    AND pma.project_id = kb_files.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can update KB files if they have edit access to the project
CREATE POLICY "KB Files - Member update access" ON kb_files
FOR UPDATE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = kb_files.workspace_id
    AND pma.project_id = kb_files.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can delete KB files if they have edit access to the project
CREATE POLICY "KB Files - Member delete access" ON kb_files
FOR DELETE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = kb_files.workspace_id
    AND pma.project_id = kb_files.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- RLS Policies for call_analysis_results table
-- Policy: Workspace owners/admins have full access to all call analysis results in their workspaces
CREATE POLICY "Call Analysis Results - Owner and admin access" ON call_analysis_results
FOR ALL TO authenticated
USING (
  workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
);

-- Policy: Members can view call analysis results based on project access level
CREATE POLICY "Call Analysis Results - Member view access" ON call_analysis_results
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = call_analysis_results.workspace_id
    AND pma.project_id = call_analysis_results.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level IN ('view', 'edit')
  )
);

-- Policy: Members can insert call analysis results if they have edit access to the project
CREATE POLICY "Call Analysis Results - Member insert access" ON call_analysis_results
FOR INSERT TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = call_analysis_results.workspace_id
    AND pma.project_id = call_analysis_results.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can update call analysis results if they have edit access to the project
CREATE POLICY "Call Analysis Results - Member update access" ON call_analysis_results
FOR UPDATE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = call_analysis_results.workspace_id
    AND pma.project_id = call_analysis_results.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can delete call analysis results if they have edit access to the project
CREATE POLICY "Call Analysis Results - Member delete access" ON call_analysis_results
FOR DELETE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = call_analysis_results.workspace_id
    AND pma.project_id = call_analysis_results.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- RLS Policies for project_delays table
-- Policy: Workspace owners/admins have full access to all project delays in their workspaces
CREATE POLICY "Project Delays - Owner and admin access" ON project_delays
FOR ALL TO authenticated
USING (
  workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
);

-- Policy: Members can view project delays based on project access level
CREATE POLICY "Project Delays - Member view access" ON project_delays
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_delays.workspace_id
    AND pma.project_id = project_delays.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level IN ('view', 'edit')
  )
);

-- Policy: Members can insert project delays if they have edit access to the project
CREATE POLICY "Project Delays - Member insert access" ON project_delays
FOR INSERT TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_delays.workspace_id
    AND pma.project_id = project_delays.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can update project delays if they have edit access to the project
CREATE POLICY "Project Delays - Member update access" ON project_delays
FOR UPDATE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_delays.workspace_id
    AND pma.project_id = project_delays.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can delete project delays if they have edit access to the project
CREATE POLICY "Project Delays - Member delete access" ON project_delays
FOR DELETE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_delays.workspace_id
    AND pma.project_id = project_delays.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- RLS Policies for project_rfis table
-- Policy: Workspace owners/admins have full access to all project RFIs in their workspaces
CREATE POLICY "Project RFIs - Owner and admin access" ON project_rfis
FOR ALL TO authenticated
USING (
  workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
);

-- Policy: Members can view project RFIs based on project access level
CREATE POLICY "Project RFIs - Member view access" ON project_rfis
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_rfis.workspace_id
    AND pma.project_id = project_rfis.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level IN ('view', 'edit')
  )
);

-- Policy: Members can insert project RFIs if they have edit access to the project
CREATE POLICY "Project RFIs - Member insert access" ON project_rfis
FOR INSERT TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_rfis.workspace_id
    AND pma.project_id = project_rfis.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can update project RFIs if they have edit access to the project
CREATE POLICY "Project RFIs - Member update access" ON project_rfis
FOR UPDATE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_rfis.workspace_id
    AND pma.project_id = project_rfis.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can delete project RFIs if they have edit access to the project
CREATE POLICY "Project RFIs - Member delete access" ON project_rfis
FOR DELETE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_rfis.workspace_id
    AND pma.project_id = project_rfis.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- RLS Policies for project_safety_incidents table
-- Policy: Workspace owners/admins have full access to all project safety incidents in their workspaces
CREATE POLICY "Project Safety Incidents - Owner and admin access" ON project_safety_incidents
FOR ALL TO authenticated
USING (
  workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
);

-- Policy: Members can view project safety incidents based on project access level
CREATE POLICY "Project Safety Incidents - Member view access" ON project_safety_incidents
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_safety_incidents.workspace_id
    AND pma.project_id = project_safety_incidents.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level IN ('view', 'edit')
  )
);

-- Policy: Members can insert project safety incidents if they have edit access to the project
CREATE POLICY "Project Safety Incidents - Member insert access" ON project_safety_incidents
FOR INSERT TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_safety_incidents.workspace_id
    AND pma.project_id = project_safety_incidents.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can update project safety incidents if they have edit access to the project
CREATE POLICY "Project Safety Incidents - Member update access" ON project_safety_incidents
FOR UPDATE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_safety_incidents.workspace_id
    AND pma.project_id = project_safety_incidents.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can delete project safety incidents if they have edit access to the project
CREATE POLICY "Project Safety Incidents - Member delete access" ON project_safety_incidents
FOR DELETE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_safety_incidents.workspace_id
    AND pma.project_id = project_safety_incidents.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- RLS Policies for project_actions table
-- Policy: Workspace owners/admins have full access to all project actions in their workspaces
CREATE POLICY "Project Actions - Owner and admin access" ON project_actions
FOR ALL TO authenticated
USING (
  workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
);

-- Policy: Members can view project actions based on project access level
CREATE POLICY "Project Actions - Member view access" ON project_actions
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_actions.workspace_id
    AND pma.project_id = project_actions.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level IN ('view', 'edit')
  )
);

-- Policy: Members can insert project actions if they have edit access to the project
CREATE POLICY "Project Actions - Member insert access" ON project_actions
FOR INSERT TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_actions.workspace_id
    AND pma.project_id = project_actions.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can update project actions if they have edit access to the project
CREATE POLICY "Project Actions - Member update access" ON project_actions
FOR UPDATE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_actions.workspace_id
    AND pma.project_id = project_actions.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can delete project actions if they have edit access to the project
CREATE POLICY "Project Actions - Member delete access" ON project_actions
FOR DELETE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_actions.workspace_id
    AND pma.project_id = project_actions.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- RLS Policies for project_issues table
-- Policy: Workspace owners/admins have full access to all project issues in their workspaces
CREATE POLICY "Project Issues - Owner and admin access" ON project_issues
FOR ALL TO authenticated
USING (
  workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
);

-- Policy: Members can view project issues based on project access level
CREATE POLICY "Project Issues - Member view access" ON project_issues
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_issues.workspace_id
    AND pma.project_id = project_issues.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level IN ('view', 'edit')
  )
);

-- Policy: Members can insert project issues if they have edit access to the project
CREATE POLICY "Project Issues - Member insert access" ON project_issues
FOR INSERT TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_issues.workspace_id
    AND pma.project_id = project_issues.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can update project issues if they have edit access to the project
CREATE POLICY "Project Issues - Member update access" ON project_issues
FOR UPDATE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_issues.workspace_id
    AND pma.project_id = project_issues.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can delete project issues if they have edit access to the project
CREATE POLICY "Project Issues - Member delete access" ON project_issues
FOR DELETE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_issues.workspace_id
    AND pma.project_id = project_issues.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- RLS Policies for project_contacts table
-- Policy: Workspace owners/admins have full access to all project contacts in their workspaces
CREATE POLICY "Project Contacts - Owner and admin access" ON project_contacts
FOR ALL TO authenticated
USING (
  workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
);

-- Policy: Members can view project contacts based on project access level
CREATE POLICY "Project Contacts - Member view access" ON project_contacts
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_contacts.workspace_id
    AND pma.project_id = project_contacts.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level IN ('view', 'edit')
  )
);

-- Policy: Members can insert project contacts if they have edit access to the project
CREATE POLICY "Project Contacts - Member insert access" ON project_contacts
FOR INSERT TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_contacts.workspace_id
    AND pma.project_id = project_contacts.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can update project contacts if they have edit access to the project
CREATE POLICY "Project Contacts - Member update access" ON project_contacts
FOR UPDATE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_contacts.workspace_id
    AND pma.project_id = project_contacts.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can delete project contacts if they have edit access to the project
CREATE POLICY "Project Contacts - Member delete access" ON project_contacts
FOR DELETE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_contacts.workspace_id
    AND pma.project_id = project_contacts.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- RLS Policies for dpr_call_settings table
-- Policy: Workspace owners/admins have full access to all DPR call settings in their workspaces
CREATE POLICY "DPR Call Settings - Owner and admin access" ON dpr_call_settings
FOR ALL TO authenticated
USING (
  workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
);

-- Policy: Members can view DPR call settings based on project access level
CREATE POLICY "DPR Call Settings - Member view access" ON dpr_call_settings
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = dpr_call_settings.workspace_id
    AND pma.project_id = dpr_call_settings.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level IN ('view', 'edit')
  )
);

-- Policy: Members can insert DPR call settings if they have edit access to the project
CREATE POLICY "DPR Call Settings - Member insert access" ON dpr_call_settings
FOR INSERT TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = dpr_call_settings.workspace_id
    AND pma.project_id = dpr_call_settings.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can update DPR call settings if they have edit access to the project
CREATE POLICY "DPR Call Settings - Member update access" ON dpr_call_settings
FOR UPDATE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = dpr_call_settings.workspace_id
    AND pma.project_id = dpr_call_settings.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can delete DPR call settings if they have edit access to the project
CREATE POLICY "DPR Call Settings - Member delete access" ON dpr_call_settings
FOR DELETE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = dpr_call_settings.workspace_id
    AND pma.project_id = dpr_call_settings.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- RLS Policies for project_work_progress table
-- Policy: Workspace owners/admins have full access to all project work progress in their workspaces
CREATE POLICY "Project Work Progress - Owner and admin access" ON project_work_progress
FOR ALL TO authenticated
USING (
  workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
);

-- Policy: Members can view project work progress based on project access level
CREATE POLICY "Project Work Progress - Member view access" ON project_work_progress
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_work_progress.workspace_id
    AND pma.project_id = project_work_progress.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level IN ('view', 'edit')
  )
);

-- Policy: Members can insert project work progress if they have edit access to the project
CREATE POLICY "Project Work Progress - Member insert access" ON project_work_progress
FOR INSERT TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_work_progress.workspace_id
    AND pma.project_id = project_work_progress.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can update project work progress if they have edit access to the project
CREATE POLICY "Project Work Progress - Member update access" ON project_work_progress
FOR UPDATE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_work_progress.workspace_id
    AND pma.project_id = project_work_progress.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can delete project work progress if they have edit access to the project
CREATE POLICY "Project Work Progress - Member delete access" ON project_work_progress
FOR DELETE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = project_work_progress.workspace_id
    AND pma.project_id = project_work_progress.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- RLS Policies for work_progress table
-- Policy: Workspace owners/admins have full access to all work progress in their workspaces
CREATE POLICY "Work Progress - Owner and admin access" ON work_progress
FOR ALL TO authenticated
USING (
  workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
);

-- Policy: Members can view work progress based on project access level
CREATE POLICY "Work Progress - Member view access" ON work_progress
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = work_progress.workspace_id
    AND pma.project_id = work_progress.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level IN ('view', 'edit')
  )
);

-- Policy: Members can insert work progress if they have edit access to the project
CREATE POLICY "Work Progress - Member insert access" ON work_progress
FOR INSERT TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = work_progress.workspace_id
    AND pma.project_id = work_progress.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can update work progress if they have edit access to the project
CREATE POLICY "Work Progress - Member update access" ON work_progress
FOR UPDATE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = work_progress.workspace_id
    AND pma.project_id = work_progress.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can delete work progress if they have edit access to the project
CREATE POLICY "Work Progress - Member delete access" ON work_progress
FOR DELETE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = work_progress.workspace_id
    AND pma.project_id = work_progress.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- RLS Policies for issues table
-- Policy: Workspace owners/admins have full access to all issues in their workspaces
CREATE POLICY "Issues - Owner and admin access" ON issues
FOR ALL TO authenticated
USING (
  workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
);

-- Policy: Members can view issues based on project access level
CREATE POLICY "Issues - Member view access" ON issues
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = issues.workspace_id
    AND pma.project_id = issues.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level IN ('view', 'edit')
  )
);

-- Policy: Members can insert issues if they have edit access to the project
CREATE POLICY "Issues - Member insert access" ON issues
FOR INSERT TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = issues.workspace_id
    AND pma.project_id = issues.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can update issues if they have edit access to the project
CREATE POLICY "Issues - Member update access" ON issues
FOR UPDATE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = issues.workspace_id
    AND pma.project_id = issues.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can delete issues if they have edit access to the project
CREATE POLICY "Issues - Member delete access" ON issues
FOR DELETE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = issues.workspace_id
    AND pma.project_id = issues.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- RLS Policies for site_media table
-- Policy: Workspace owners/admins have full access to all site media in their workspaces
CREATE POLICY "Site Media - Owner and admin access" ON site_media
FOR ALL TO authenticated
USING (
  workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
);

-- Policy: Members can view site media based on project access level
CREATE POLICY "Site Media - Member view access" ON site_media
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = site_media.workspace_id
    AND pma.project_id = site_media.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level IN ('view', 'edit')
  )
);

-- Policy: Members can insert site media if they have edit access to the project
CREATE POLICY "Site Media - Member insert access" ON site_media
FOR INSERT TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = site_media.workspace_id
    AND pma.project_id = site_media.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can update site media if they have edit access to the project
CREATE POLICY "Site Media - Member update access" ON site_media
FOR UPDATE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = site_media.workspace_id
    AND pma.project_id = site_media.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can delete site media if they have edit access to the project
CREATE POLICY "Site Media - Member delete access" ON site_media
FOR DELETE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = site_media.workspace_id
    AND pma.project_id = site_media.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Create RLS policies for project_member_access
CREATE POLICY "Users can view their project access"
    ON project_member_access FOR SELECT
    TO authenticated
    USING (user_id = (select auth.uid()));

-- RLS Policies for sla_settings table
-- Policy: Workspace owners/admins have full access to all SLA settings in their workspaces
CREATE POLICY "SLA Settings - Owner and admin access" ON sla_settings
FOR ALL TO authenticated
USING (
  workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
);

-- Policy: Members can view SLA settings based on project access level (if project_id is set)
CREATE POLICY "SLA Settings - Member view access" ON sla_settings
FOR SELECT TO authenticated
USING (
  -- If project_id is NULL, only owners/admins can view workspace-level settings
  project_id IS NULL AND workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
  OR
  -- If project_id is set, check project access
  (project_id IS NOT NULL AND EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = sla_settings.workspace_id
    AND pma.project_id = sla_settings.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level IN ('view', 'edit')
  ))
);

-- Policy: Members can insert SLA settings if they have edit access to the project (or are owner/admin for workspace-level)
CREATE POLICY "SLA Settings - Member insert access" ON sla_settings
FOR INSERT TO authenticated
WITH CHECK (
  -- Workspace-level settings (project_id is NULL) - only owners/admins
  (project_id IS NULL AND workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  ))
  OR
  -- Project-level settings - check project access
  (project_id IS NOT NULL AND EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = sla_settings.workspace_id
    AND pma.project_id = sla_settings.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  ))
);

-- Policy: Members can update SLA settings if they have edit access to the project (or are owner/admin for workspace-level)
CREATE POLICY "SLA Settings - Member update access" ON sla_settings
FOR UPDATE TO authenticated
USING (
  -- Workspace-level settings (project_id is NULL) - only owners/admins
  (project_id IS NULL AND workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  ))
  OR
  -- Project-level settings - check project access
  (project_id IS NOT NULL AND EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = sla_settings.workspace_id
    AND pma.project_id = sla_settings.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  ))
);

-- Policy: Members can delete SLA settings if they have edit access to the project (or are owner/admin for workspace-level)
CREATE POLICY "SLA Settings - Member delete access" ON sla_settings
FOR DELETE TO authenticated
USING (
  -- Workspace-level settings (project_id is NULL) - only owners/admins
  (project_id IS NULL AND workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  ))
  OR
  -- Project-level settings - check project access
  (project_id IS NOT NULL AND EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = sla_settings.workspace_id
    AND pma.project_id = sla_settings.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  ))
);

-- RLS Policies for subtasks table
-- Policy: Workspace owners/admins have full access to all subtasks in their workspaces
CREATE POLICY "Subtasks - Owner and admin access" ON subtasks
FOR ALL TO authenticated
USING (
  workspace_id IN (
    SELECT workspace_id FROM workspace_members
    WHERE user_id = (select auth.uid())
    AND role IN ('owner', 'admin')
  )
);

-- Policy: Members can view subtasks based on project access level
CREATE POLICY "Subtasks - Member view access" ON subtasks
FOR SELECT TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = subtasks.workspace_id
    AND pma.project_id = subtasks.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level IN ('view', 'edit')
  )
);

-- Policy: Members can insert subtasks if they have edit access to the project
CREATE POLICY "Subtasks - Member insert access" ON subtasks
FOR INSERT TO authenticated
WITH CHECK (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = subtasks.workspace_id
    AND pma.project_id = subtasks.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can update subtasks if they have edit access to the project
CREATE POLICY "Subtasks - Member update access" ON subtasks
FOR UPDATE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = subtasks.workspace_id
    AND pma.project_id = subtasks.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Policy: Members can delete subtasks if they have edit access to the project
CREATE POLICY "Subtasks - Member delete access" ON subtasks
FOR DELETE TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM project_member_access pma
    WHERE pma.workspace_id = subtasks.workspace_id
    AND pma.project_id = subtasks.project_id
    AND pma.user_id = (select auth.uid())
    AND pma.access_level = 'edit'
  )
);

-- Create storage bucket for Project files
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'kb-files',
  'kb-files',
  false,
  5242880, -- 5MB limit
  ARRAY['application/pdf']
) ON CONFLICT (id) DO NOTHING;

-- Create storage bucket for Call Recordings
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'call-recordings',
  'call-recordings',
  false,
  52428800, -- 50MB limit for audio files
  ARRAY['audio/mp3', 'audio/wav']
) ON CONFLICT (id) DO NOTHING;

-- Create storage bucket for Transcripts
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'transcripts',
  'transcripts',
  false,
  10485760, -- 10MB limit for text files
  ARRAY['text/plain']
) ON CONFLICT (id) DO NOTHING;

-- Create storage bucket for Site Media
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'site-media',
  'site-media',
  false,
  52428800, -- 50MB limit for image and video files
  ARRAY['image/jpeg', 'image/png', 'image/heic', 'image/heif', 'video/mp4', 'video/quicktime', 'video/x-msvideo']
) ON CONFLICT (id) DO NOTHING;

-- Create storage bucket for project finances
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'project-finances',
  'project-finances', 
  false,
  10485760, -- 10MB limit
  ARRAY['image/jpeg', 'image/png', 'image/heic', 'image/heif', 'application/pdf']
) ON CONFLICT (id) DO NOTHING;

-- Create storage bucket for issue attachments
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'issue-attachments',
  'issue-attachments',
  false,
  26214400, -- 25MB limit
  ARRAY[
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'text/plain',
    'text/csv',
    'application/zip',
    'image/jpeg',
    'image/png'
  ]
) ON CONFLICT (id) DO NOTHING;

-- Create storage bucket for issue photos
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'issue-photos',
  'issue-photos',
  false,
  31457280, -- 30MB limit
  ARRAY['image/jpeg', 'image/png', 'image/heic', 'image/heif']
) ON CONFLICT (id) DO NOTHING;

-- Storage policies for project-finances bucket
-- File path structure: {workspace_id}/{project_id}/cashflow_{timestamp}-{random}.{ext}

-- Policy: Users can upload files to their workspace's project finance folders
CREATE POLICY "Project Finances - Upload" ON storage.objects
FOR INSERT TO authenticated
WITH CHECK (
  bucket_id = 'project-finances' AND
  (
    -- Owner/Admin users have full access
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    -- Members with edit access can upload
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level = 'edit'
    )
  )
);

-- Policy: Users can view finance files from their workspace projects
CREATE POLICY "Project Finances - View" ON storage.objects
FOR SELECT TO authenticated
USING (
  bucket_id = 'project-finances' AND
  (
    -- Owner/Admin users have full access
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    -- Members with view or edit access can view
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level IN ('view', 'edit')
    )
  )
);

-- Policy: Users can update finance files in their workspace projects
CREATE POLICY "Project Finances - Update" ON storage.objects
FOR UPDATE TO authenticated
USING (
  bucket_id = 'project-finances' AND
  (
    -- Owner/Admin users have full access
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    -- Members with edit access can update
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level = 'edit'
    )
  )
);

-- Policy: Users can delete finance files from their workspace projects
CREATE POLICY "Project Finances - Delete" ON storage.objects
FOR DELETE TO authenticated
USING (
  bucket_id = 'project-finances' AND
  (
    -- Owner/Admin users have full access
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    -- Members with edit access can delete
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level = 'edit'
    )
  )
);

-- Storage policies for kb-files bucket
-- File path structure: {workspace_id}/{project_id}/filename.ext

-- Policy: Users can upload KB files to their workspace's project folders
CREATE POLICY "KB Files - Upload" ON storage.objects
FOR INSERT TO authenticated
WITH CHECK (
  bucket_id = 'kb-files' AND
  (
    -- Owner/Admin users have full access
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    -- Members with edit access can upload
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level = 'edit'
    )
  )
);

-- Policy: Users can view KB files from their workspace projects
CREATE POLICY "KB Files - View" ON storage.objects
FOR SELECT TO authenticated
USING (
  bucket_id = 'kb-files' AND
  (
    -- Owner/Admin users have full access
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    -- Members with view or edit access can view
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level IN ('view', 'edit')
    )
  )
);

-- Policy: Users can update KB files in their workspace projects
CREATE POLICY "KB Files - Update" ON storage.objects
FOR UPDATE TO authenticated
USING (
  bucket_id = 'kb-files' AND
  (
    -- Owner/Admin users have full access
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    -- Members with edit access can update
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level = 'edit'
    )
  )
);

-- Policy: Users can delete KB files from their workspace projects
CREATE POLICY "KB Files - Delete" ON storage.objects
FOR DELETE TO authenticated
USING (
  bucket_id = 'kb-files' AND
  (
    -- Owner/Admin users have full access
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    -- Members with edit access can delete
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level = 'edit'
    )
  )
);

-- Storage policies for call-recordings bucket
-- File path structure: {workspace_id}/{project_id}/recording_{timestamp}-{random}.{ext}

-- Policy: Users can upload call recording files to their workspace's project folders
CREATE POLICY "Call Recordings - Upload" ON storage.objects
FOR INSERT TO authenticated
WITH CHECK (
  bucket_id = 'call-recordings' AND
  (
    -- Owner/Admin users have full access
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    -- Members with edit access can upload
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level = 'edit'
    )
  )
);

-- Policy: Users can view call recording files from their workspace projects
CREATE POLICY "Call Recordings - View" ON storage.objects
FOR SELECT TO authenticated
USING (
  bucket_id = 'call-recordings' AND
  (
    -- Owner/Admin users have full access
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    -- Members with view or edit access can view
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level IN ('view', 'edit')
    )
  )
);

-- Policy: Users can update call recording files in their workspace projects
CREATE POLICY "Call Recordings - Update" ON storage.objects
FOR UPDATE TO authenticated
USING (
  bucket_id = 'call-recordings' AND
  (
    -- Owner/Admin users have full access
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    -- Members with edit access can update
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level = 'edit'
    )
  )
);

-- Policy: Users can delete call recording files from their workspace projects
CREATE POLICY "Call Recordings - Delete" ON storage.objects
FOR DELETE TO authenticated
USING (
  bucket_id = 'call-recordings' AND
  (
    -- Owner/Admin users have full access
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    -- Members with edit access can delete
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level = 'edit'
    )
  )
);

-- Storage policies for transcripts bucket
-- File path structure: {workspace_id}/{project_id}/transcript_{timestamp}-{random}.txt

-- Policy: Users can upload transcript files to their workspace's project folders
CREATE POLICY "Transcripts - Upload" ON storage.objects
FOR INSERT TO authenticated
WITH CHECK (
  bucket_id = 'transcripts' AND
  (
    -- Owner/Admin users have full access
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    -- Members with edit access can upload
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level = 'edit'
    )
  )
);

-- Policy: Users can view transcript files from their workspace projects
CREATE POLICY "Transcripts - View" ON storage.objects
FOR SELECT TO authenticated
USING (
  bucket_id = 'transcripts' AND
  (
    -- Owner/Admin users have full access
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    -- Members with view or edit access can view
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level IN ('view', 'edit')
    )
  )
);

-- Policy: Users can update transcript files in their workspace projects
CREATE POLICY "Transcripts - Update" ON storage.objects
FOR UPDATE TO authenticated
USING (
  bucket_id = 'transcripts' AND
  (
    -- Owner/Admin users have full access
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    -- Members with edit access can update
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level = 'edit'
    )
  )
);

-- Policy: Users can delete transcript files from their workspace projects
CREATE POLICY "Transcripts - Delete" ON storage.objects
FOR DELETE TO authenticated
USING (
  bucket_id = 'transcripts' AND
  (
    -- Owner/Admin users have full access
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    -- Members with edit access can delete
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level = 'edit'
    )
  )
);

-- Storage policies for site-media bucket
-- File path structure: {workspace_id}/{project_id}/media_{timestamp}-{random}.{ext}

-- Policy: Users can upload site media files to their workspace's project folders
CREATE POLICY "Site Media - Upload" ON storage.objects
FOR INSERT TO authenticated
WITH CHECK (
  bucket_id = 'site-media' AND
  (
    -- Owner/Admin users have full access
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    -- Members with edit access can upload
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level = 'edit'
    )
  )
);

-- Policy: Users can view site media files from their workspace projects
CREATE POLICY "Site Media - View" ON storage.objects
FOR SELECT TO authenticated
USING (
  bucket_id = 'site-media' AND
  (
    -- Owner/Admin users have full access
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    -- Members with view or edit access can view
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level IN ('view', 'edit')
    )
  )
);

-- Policy: Users can update site media files in their workspace projects
CREATE POLICY "Site Media - Update" ON storage.objects
FOR UPDATE TO authenticated
USING (
  bucket_id = 'site-media' AND
  (
    -- Owner/Admin users have full access
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    -- Members with edit access can update
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level = 'edit'
    )
  )
);

-- Policy: Users can delete site media files from their workspace projects
CREATE POLICY "Site Media - Delete" ON storage.objects
FOR DELETE TO authenticated
USING (
  bucket_id = 'site-media' AND
  (
    -- Owner/Admin users have full access
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    -- Members with edit access can delete
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level = 'edit'
    )
  )
);

-- Storage policies for issue attachments bucket
-- File path structure: {workspace_id}/{project_id}/attachments/{filename}

CREATE POLICY "Issue Attachments - Upload" ON storage.objects
FOR INSERT TO authenticated
WITH CHECK (
  bucket_id = 'issue-attachments' AND
  (
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level = 'edit'
    )
  )
);

CREATE POLICY "Issue Attachments - View" ON storage.objects
FOR SELECT TO authenticated
USING (
  bucket_id = 'issue-attachments' AND
  (
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level IN ('view', 'edit')
    )
  )
);

CREATE POLICY "Issue Attachments - Update" ON storage.objects
FOR UPDATE TO authenticated
USING (
  bucket_id = 'issue-attachments' AND
  (
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level = 'edit'
    )
  )
);

CREATE POLICY "Issue Attachments - Delete" ON storage.objects
FOR DELETE TO authenticated
USING (
  bucket_id = 'issue-attachments' AND
  (
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level = 'edit'
    )
  )
);

-- Storage policies for issue photos bucket
-- File path structure: {workspace_id}/{project_id}/photos/{filename}

CREATE POLICY "Issue Photos - Upload" ON storage.objects
FOR INSERT TO authenticated
WITH CHECK (
  bucket_id = 'issue-photos' AND
  (
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level = 'edit'
    )
  )
);

CREATE POLICY "Issue Photos - View" ON storage.objects
FOR SELECT TO authenticated
USING (
  bucket_id = 'issue-photos' AND
  (
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level IN ('view', 'edit')
    )
  )
);

CREATE POLICY "Issue Photos - Update" ON storage.objects
FOR UPDATE TO authenticated
USING (
  bucket_id = 'issue-photos' AND
  (
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level = 'edit'
    )
  )
);

CREATE POLICY "Issue Photos - Delete" ON storage.objects
FOR DELETE TO authenticated
USING (
  bucket_id = 'issue-photos' AND
  (
    (storage.foldername(name))[1]::uuid IN (
      SELECT workspace_id FROM workspace_members
      WHERE user_id = (select auth.uid())
      AND role IN ('owner', 'admin')
    )
    OR
    EXISTS (
      SELECT 1 FROM project_member_access pma
      WHERE pma.workspace_id = (storage.foldername(name))[1]::uuid
      AND pma.project_id = (storage.foldername(name))[2]::uuid
      AND pma.user_id = (select auth.uid())
      AND pma.access_level = 'edit'
    )
  )
);

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
CREATE TRIGGER update_workspaces_updated_at
    BEFORE UPDATE ON workspaces
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_subscriptions_updated_at
    BEFORE UPDATE ON subscriptions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_profiles_updated_at
    BEFORE UPDATE ON profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agents_updated_at
    BEFORE UPDATE ON agents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_phone_numbers_updated_at
    BEFORE UPDATE ON phone_numbers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_phone_pools_updated_at
    BEFORE UPDATE ON phone_pools
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_phone_mappings_updated_at
    BEFORE UPDATE ON agent_phone_mappings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_call_logs_updated_at
    BEFORE UPDATE ON call_logs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tasks_updated_at
    BEFORE UPDATE ON tasks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_contacts_updated_at
    BEFORE UPDATE ON contacts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_projects_updated_at
    BEFORE UPDATE ON projects
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_kb_files_updated_at
    BEFORE UPDATE ON kb_files
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create trigger for updated_at
CREATE TRIGGER update_call_analysis_results_updated_at
    BEFORE UPDATE ON call_analysis_results
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_project_schedules_updated_at
    BEFORE UPDATE ON project_schedules
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_project_schedule_dependencies_updated_at
    BEFORE UPDATE ON project_schedule_dependencies
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_project_budgets_updated_at
    BEFORE UPDATE ON project_budgets
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_material_tracking_updated_at
    BEFORE UPDATE ON material_tracking
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_equipment_tracking_updated_at
    BEFORE UPDATE ON equipment_tracking
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_labor_tracking_updated_at
    BEFORE UPDATE ON labor_tracking
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_vendors_updated_at
    BEFORE UPDATE ON vendors
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_vendor_rates_updated_at
    BEFORE UPDATE ON vendor_rates
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_pricing_plans_updated_at
    BEFORE UPDATE ON pricing_plans
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_project_alerts_updated_at
    BEFORE UPDATE ON project_alerts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ai_recommendations_updated_at
    BEFORE UPDATE ON ai_recommendations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_project_usage_updated_at
    BEFORE UPDATE ON project_usage
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_project_delays_updated_at
    BEFORE UPDATE ON project_delays
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_project_rfis_updated_at
    BEFORE UPDATE ON project_rfis
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_project_safety_incidents_updated_at
    BEFORE UPDATE ON project_safety_incidents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_project_actions_updated_at
    BEFORE UPDATE ON project_actions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_project_issues_updated_at
    BEFORE UPDATE ON project_issues
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_project_contacts_updated_at
    BEFORE UPDATE ON project_contacts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_dpr_call_settings_updated_at
    BEFORE UPDATE ON dpr_call_settings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_project_work_progress_updated_at
    BEFORE UPDATE ON project_work_progress
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_work_progress_updated_at
    BEFORE UPDATE ON work_progress
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_issues_updated_at
    BEFORE UPDATE ON issues
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_site_media_updated_at
    BEFORE UPDATE ON site_media
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sla_settings_updated_at
    BEFORE UPDATE ON sla_settings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_subtasks_updated_at
    BEFORE UPDATE ON subtasks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Project Usage Functions and Triggers
-- These functions insert entries into project_usage table (unlike usage table which uses incremental updates)

-- Function to insert call minutes usage into project_usage table
CREATE OR REPLACE FUNCTION public.insert_project_call_minutes_usage()
RETURNS TRIGGER 
SET search_path = public, pg_catalog
LANGUAGE plpgsql
AS $$
BEGIN
  -- Only insert if we have both workspace_id and project_id, and bill_duration is not null
  IF NEW.workspace_id IS NOT NULL AND NEW.project_id IS NOT NULL AND NEW.bill_duration IS NOT NULL THEN
    INSERT INTO public.project_usage (
      workspace_id, 
      project_id, 
      call_minutes_used, 
      created_at, 
      updated_at
    )
    VALUES (
      NEW.workspace_id, 
      NEW.project_id, 
      CEIL(NEW.bill_duration / 60.0)::INTEGER, 
      now(), 
      now()
    );
  END IF;

  RETURN NEW;
END;
$$;

-- Trigger on call_logs INSERT to track call minutes per project
CREATE TRIGGER trg_insert_project_call_minutes_usage
AFTER INSERT ON call_logs
FOR EACH ROW
WHEN (NEW.bill_duration IS NOT NULL AND NEW.workspace_id IS NOT NULL AND NEW.project_id IS NOT NULL)
EXECUTE FUNCTION insert_project_call_minutes_usage();

-- Insert sample pricing plans
INSERT INTO pricing_plans (
    id,
    name,
    min_project_size,
    max_project_size,
    base_fee,
    billing_period,
    description,
    features,
    is_active,
    call_minutes_included,
    phone_numbers_included,
    tokens_included
) VALUES 
(
    gen_random_uuid(),
    'Trial',
    0,
    100000000,
    0.00,
    'trial',
    'One-time free trial for new workspaces',
    '[]',
    true,
    200,
    1,
    1000000
),
-- Starter Plan
(
    gen_random_uuid(),
    'Starter',
    0,
    1000000,
    75.00,
    'monthly',
    'Perfect for independent contractors and solopreneurs',
    '[]',
    true,
    200,
    1,
    2000000
),
-- Standard Plan
(
    gen_random_uuid(),
    'Standard',
    0,
    5000000,
    150.00,
    'monthly',
    'Ideal for growing construction teams',
    '[]',
    true,
    400,
    2,
    4000000
),
-- Pro Plan
(
    gen_random_uuid(),
    'Pro',
    0,
    10000000,
    400.00,
    'monthly',
    'For established construction companies and large site project managers',
    '[]',
    true,
    900,
    4,
    9000000
)
ON CONFLICT (id) DO NOTHING;