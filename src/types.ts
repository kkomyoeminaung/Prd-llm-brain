/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export enum RegionType {
  REASONING = 'Reasoning',
  LANGUAGE = 'Language',
  MATH = 'Mathematics',
  MEMORY = 'Memory',
  CODE = 'Code',
  VISION = 'Vision',
  MOTOR = 'Motor',
  EMOTIONAL = 'Emotional'
}

export interface BrainState {
  activeRegions: RegionType[];
  confidence: number;
  plasticityLevel: number;
  isProcessing: boolean;
  criticCorrectionCount: number;
  lastMessage?: string;
}

export const REGIONS_CONFIG = {
  [RegionType.REASONING]: { color: '#8EA695', icon: 'Brain' }, // Sage
  [RegionType.LANGUAGE]: { color: '#BDA78F', icon: 'MessageSquare' }, // Sand
  [RegionType.MATH]: { color: '#D4A373', icon: 'Hash' }, // Terracotta
  [RegionType.MEMORY]: { color: '#E8D5C4', icon: 'Database' }, // Clay
  [RegionType.CODE]: { color: '#77877E', icon: 'Code' }, // Dark Sage
  [RegionType.VISION]: { color: '#E9E0CE', icon: 'Eye' }, // Cream
  [RegionType.MOTOR]: { color: '#A98467', icon: 'Activity' }, // Walnut
  [RegionType.EMOTIONAL]: { color: '#DD9081', icon: 'Heart' }, // Muted Red
};
