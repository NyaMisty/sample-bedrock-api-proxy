export interface ModelMapping {
  anthropic_model_id: string;
  bedrock_model_id: string;
  source: 'default' | 'custom' | 'override';
  default_bedrock_model_id?: string;
  updated_at?: number;
}

export interface ModelMappingCreate {
  anthropic_model_id: string;
  bedrock_model_id: string;
}

export interface ModelMappingUpdate {
  bedrock_model_id: string;
}

export interface ModelMappingListResponse {
  items: ModelMapping[];
  count: number;
}
